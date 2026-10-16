"""Logged transport binding for the pinned, original WorldMind plugin.

Only LLM transport is replaced. Native prompts, parsers, discrimination,
reflection, storage, retrieval and refinement remain unchanged. The plugin
needs a host that supplies predictions; it is not itself a standalone agent.
"""
from pathlib import Path
import sys
import types

PLUGIN = Path(__file__).resolve().parents[2]/'vendor/WorldMind/Plugin'
if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))

# This pinned package initializer re-exports a missing parse_llm_output name.
# Load the unchanged implementation submodules through a package namespace,
# bypassing only that broken public-export initializer.
if 'worldmind_plugin' not in sys.modules:
    package = types.ModuleType('worldmind_plugin')
    package.__path__ = [str(PLUGIN/'worldmind_plugin')]
    sys.modules['worldmind_plugin'] = package

from worldmind_plugin.config import WorldMindConfig
from worldmind_plugin.core import (ProcessExperienceModule, GoalExperienceModule,
                                  ExperienceRetrievalModule, ProcessTrajectoryStep,
                                  GoalTrajectoryStep)
from worldmind_plugin import utils


class NativeWorldMind:
    def __init__(self, client, directory, prefix):
        self.client = client
        self.prefix = prefix
        self.tag = 'call'
        self.errors = []
        self.discriminations = []
        self.config = WorldMindConfig(
            api_key='unused-transport-binding', is_multimodal=False,
            discriminator_model=client.model, reflector_model=client.model,
            extractor_model=client.model, refiner_model=client.model,
            save_path=str(directory), detailed_output=False)
        self.process = ProcessExperienceModule(self.config)
        self.goal = GoalExperienceModule(self.config)
        self.retrieval = ExperienceRetrievalModule(self.config)
        self._bind(self.process.discriminator.llm_client, 'discriminate')
        self._bind(self.process.reflector.llm_client, 'reflect')
        self._bind(self.goal.extractor, 'extract_goal')
        if self.retrieval.refiner:
            self._bind(self.retrieval.refiner.llm_client, 'refine')
        original = self.process.discriminator.discriminate

        def recorded_discrimination(*args, **kwargs):
            result = original(*args, **kwargs)
            self.discriminations.append(dict(tag=self.tag, result=result))
            return result

        self.process.discriminator.discriminate = recorded_discrimination

    def _bind(self, native_client, role):
        if native_client is None:
            raise RuntimeError(f'Native {role} client was not initialized')

        def call(messages):
            try:
                return self.client.chat(messages, temperature=native_client.temperature,
                    max_tokens=native_client.max_tokens, tag=f'{self.prefix}/{self.tag}/{role}')
            except Exception as exc:
                # Native modules catch transport exceptions. Keep that native
                # behavior but let the runner detect incomplete computation.
                self.errors.append(dict(tag=self.tag, role=role, error=str(exc)))
                raise

        native_client._call_api = call

    def check_transport(self):
        if self.errors:
            raise RuntimeError('Native WorldMind call did not complete: '+self.errors[-1]['error'])

    def update(self, goal, event, history):
        step = ProcessTrajectoryStep(observation=event['observation'], action=event['action'],
            predicted_state=event['prediction']['prediction'], env_feedback=event['observation'])
        result = self.process.process_single_step(task_instruction=goal, step=step,
            action_history=history, state_before=event['state'])
        self.check_transport()
        return result

    def retrieve(self, goal):
        self.retrieval.reload_experiences()
        result = self.retrieval.retrieve(goal)
        self.check_transport()
        result['similarity_backend'] = ('sentence_transformer' if utils._sentence_model is not None
                                        else 'native_word_overlap_fallback')
        return result
