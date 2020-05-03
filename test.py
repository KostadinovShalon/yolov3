from yolov3.utils import parser
from yolov3.test import test


parser = parser.get_parser_from_arguments()
test(parser)
