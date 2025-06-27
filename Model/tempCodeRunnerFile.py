from losses.cbdice_loss import SoftcbDiceLoss  

model = UNetBiFPN()

criterion = SoftcbDiceLoss()

# Example:
# predictions = model(inputs)
# loss = criterion(predictions, targets)
