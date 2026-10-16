import unittest
from wm_security.outcomes import move_outcome


class MoveOutcomeTests(unittest.TestCase):
    def test_preservation_comment_does_not_negate_success(self):
        text = ('Prediction: You move the ceramic cup to the fridge. '
                'Moving an object does not change its state of matter.')
        self.assertIs(move_outcome(text), True)

    def test_attempt_with_closed_door_is_failure(self):
        text = ('Prediction: You try to move the cup to the freezer, '
                'but the freezer door is closed. You need to open it first.')
        self.assertIs(move_outcome(text), False)

    def test_explicit_rejection(self):
        self.assertIs(move_outcome("Prediction: That can't be moved there, because the fridge isn't open."), False)

    def test_room_description_requires_review(self):
        self.assertIsNone(move_outcome('Prediction: This room is called the kitchen. The freezer contains a pot.'))


if __name__ == '__main__':
    unittest.main()
