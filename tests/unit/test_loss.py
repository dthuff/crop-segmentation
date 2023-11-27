import unittest
from crop_segmentation.loss import MulticlassDiceLoss
import torch


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.img_size = 16
        self.n_classes = 2

    def test_MulticlassDiceLoss(self):
        # ARRANGE
        dice_loss_fn = MulticlassDiceLoss(num_classes=self.n_classes, softmax_dim=1)
        y_true = torch.zeros((self.batch_size, 1,  self.img_size, self.img_size))
        y_pred = torch.zeros((self.batch_size, self.n_classes, self.img_size, self.img_size))

        y_true[0, 0, :5, :5] = 1
        y_pred[0, 1, :5, :5] = 1

        # ACT
        loss_value = dice_loss_fn(y_true, y_pred)

        # ASSERT
        torch.testing.assert_close(loss_value, torch.tensor(-1.0))


if __name__ == '__main__':
    unittest.main()
