import utils
from keras.models import Model, load_model

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='test model on stage1 test')
    parser.add_argument('--model_name', required=True,
                        metavar="name of the model to train",
                        help='Should be one of \"RCNN\" or \"UNet\"')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--csv', required=False,
                        metavar="/path/to/submission.csv",
                        help="Path to submission csv")
    args = parser.parse_args()
    result = None
    print (args.model_name, args.weights)
    if args.model_name.lower() == 'rcnn':
        model = utils.load_rcnn(args.weights)
        result = utils.test_prediction_on_stage1(model, 'rcnn', args.pathname)
    elif args.model_name.lower() == "unet":
        print("helo?")
        model = load_model(args.weights, custom_objects={'dice_coef': utils.dice_coef})
        print("helo?")
        result = utils.test_prediction_on_stage1(model, 'unet', args.pathname)
    else:
        raise Exception("Invalid model name")

    print ("Mean IOU calculated is: {}".format(result))
