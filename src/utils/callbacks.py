"""
Custom callbacks for training and evaluation.
"""

import time
import logging
from mindspore.train.callback import Callback

class LossCallback(Callback):
    """Callback for logging training loss."""
    
    def __init__(self, loss_log_file):
        """
        Initialize the callback.
        
        Args:
            loss_log_file (str): Path to the loss log file.
        """
        super(LossCallback, self).__init__()
        self.loss_log_file = loss_log_file
        self.losses = []
        self.epoch = 1
        self.step = 0
        
    def on_train_epoch_begin(self, run_context):
        """Called at the beginning of each epoch."""
        cb_params = run_context.original_args()
        self.epoch = cb_params.cur_epoch_num
        self.step = 0
        
    def on_train_step_end(self, run_context):
        """Called at the end of each training step."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        self.losses.append(float(loss))
        self.step += 1
        
        # Write detailed loss to file
        with open(self.loss_log_file, 'a', encoding='utf-8') as f:
            f.write(f"epoch: {self.epoch} step: {self.step}, loss is {float(loss):.6f}\n")
        
        # Print to terminal
        print(f"epoch: {self.epoch} step: {self.step}, loss is {float(loss):.6f}")

class EpochEndCallback(Callback):
    """Callback for logging epoch end metrics."""
    
    def __init__(self, model, eval_dataset, log_file, loss_log_file):
        """
        Initialize the callback.
        
        Args:
            model: The model to evaluate.
            eval_dataset: The evaluation dataset.
            log_file (str): Path to the main log file.
            loss_log_file (str): Path to the loss log file.
        """
        super(EpochEndCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.epoch_time = time.time()
        self.log_file = log_file
        self.loss_log_file = loss_log_file
        self.losses = []
        
    def on_train_step_end(self, run_context):
        """Called at the end of each training step."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        self.losses.append(float(loss))
        
    def on_train_epoch_end(self, run_context):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.epoch_time
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        
        # Calculate average loss for this epoch
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        self.losses = []  # Reset losses for next epoch
        
        # Evaluate on validation set
        result = self.model.eval(self.eval_dataset)
        accuracy = result['acc']
        
        # Prepare log message
        log_msg = f"\nEpoch {epoch_num} completed:\n"
        log_msg += f"Average loss: {avg_loss:.6f}\n"
        log_msg += f"Validation accuracy: {accuracy:.4f}\n"
        log_msg += f"Epoch time: {epoch_time:.2f} seconds\n"
        
        # Print to console
        print(log_msg)
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg)
        
        # Write epoch summary to loss log file
        with open(self.loss_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nEpoch {epoch_num} Summary:\n")
            f.write(f"Average loss: {avg_loss:.6f}\n")
            f.write(f"Validation accuracy: {accuracy:.4f}\n")
            f.write(f"Epoch time: {epoch_time:.2f} seconds\n")
            f.write("="*50 + "\n")
        
        self.epoch_time = time.time() 