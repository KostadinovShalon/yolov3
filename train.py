from yolov3.utils import parser
from yolov3.train import train


parser = parser.get_parser_from_arguments()
train(parser)
