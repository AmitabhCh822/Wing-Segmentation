import sys
import os

# Add the root directory of your project to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from losses.cbdice_loss import SoftcbDiceLoss


model = UNetBiFPN()

criterion = SoftcbDiceLoss()

# Example:
# predictions = model(inputs)
# loss = criterion(predictions, targets)
