from optparse import OptionParser
import sys
import yaml


class Parser:
    def __init__(self):
        self.parser = OptionParser()
        self.parser.add_option("-b", "--batch-size", default=64, dest="batch_size",
                               help="define the batch size for the trainloaders")
        self.parser.add_option("-e", "--epochs", default=50, dest="epochs",
                               help="define the number of cycles to use for training the data")
        self.parser.add_option("-s", "--save", action="store_true", dest="save_model",
                               help="define whether to save the trained model to file for later use")
        self.parser.add_option("-l", "--layers", default=2, dest="layers",
                               help="define the number of hidden layers. Must be between 2 and 4")
        self.parser.add_option("-o", "--optimizer", default="Adam", dest="optimizer",
                               help="define the optimizer to use during model training")
        self.parser.add_option("-r", "--rate", default=0.002, help="define the learning rate")
        self.parser.add_option("-f", "--file", dest="file", help="define the yaml file to read parameters from")

    def parseCLI(self, arguments=None):
        if arguments is None:
            arguments = sys.argv[1:]
        (options, args) = self.parser.parse_args(arguments)
        if options.file is not None:
            with open(options.file, "r") as file:
                try:
                    arg = yaml.safe_load(file)
                    arg['save_model'] = arg['save_model'] == 1
                    return arg
                except yaml.YAMLError as exc:
                    print(exc)
        return vars(options)


if __name__ == "__main__":
    parser = Parser()
    parser.options = parser.parseCLI(sys.argv[1:])
    print(parser.options)

