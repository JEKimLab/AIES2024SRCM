import sys

from util.yaml_loader import load_model_conf


def get_network(args, num_classes=100):
    """ return given network
    """
    info = load_model_conf(args.arch, args.arch_conf)

    # Linear
    if args.arch == 'p100_linear':
        from models.purchase100.linear import PurchaseClassifier
        net = PurchaseClassifier(num_classes=num_classes)
    elif args.arch == 'p100_linear_b':
        from models.purchase100.linear_b import PurchaseClassifier
        net = PurchaseClassifier(num_classes=num_classes, r=info['r'], d=info['d'])
    # Not Supported
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net
