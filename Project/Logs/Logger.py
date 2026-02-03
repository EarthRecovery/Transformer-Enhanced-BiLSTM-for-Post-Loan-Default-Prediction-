import sys


class Logger(object):
    def __init__(self, filename="Default.log", mode='a'):
        self.terminal = sys.stdout  # Original stdout
        self.log = open(filename, mode, encoding='utf-8')  # Log file

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.log.write(message)       # Write to file

    def flush(self):
        self.log.flush()

    def close(self):
        self.log.close()
