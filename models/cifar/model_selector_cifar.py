import sys

from util.yaml_loader import load_model_conf


def get_network(args, num_classes=100):
    """ return given network
    """
    print(args.arch, args.arch_conf)
    info = load_model_conf(args.arch, args.arch_conf)

    # VGG
    if args.arch == 'vgg':
        if args.learn_method == 'dpsgd':
            from models.cifar.vgg_gn import vgg
            net = vgg(num_classes=num_classes, arch=info['arch'])
        else:
            from models.cifar.vgg import vgg
            net = vgg(num_classes=num_classes, arch=info['arch'])
    elif args.arch == 'vgg_b':
        from models.cifar.vgg_b import vgg
        net = vgg(num_classes=num_classes, arch=info['arch'], r=info['r'], d=info['d'])
    # ResNet
    elif args.arch == 'resnet':
        if args.learn_method == 'dpsgd':
            from models.cifar.resnet_gn import resnet
            net = resnet(num_classes=num_classes, arch=info['arch'])
        else:
            from models.cifar.resnet import resnet
            net = resnet(num_classes=num_classes, arch=info['arch'])
    elif args.arch == 'resnet_b':
        from models.cifar.resnet_b import resnet
        net = resnet(num_classes=num_classes, arch=info['arch'], r=info['r'], d=info['d'])
    # MobileNetV3
    elif args.arch == 'mobilenetv3':
        from models.cifar.mobilenetv3 import mobilenetv3
        net = mobilenetv3(num_classes=num_classes, arch=info['arch'], width_mult=info['width_mult'])
    elif args.arch == 'mobilenetv3_b':
        from models.cifar.mobilenetv3_b import mobilenetv3
        net = mobilenetv3(num_classes=num_classes, arch=info['arch'], width_mult=info['width_mult'], r=info['r'], d=info['d'])
    # DenseNet
    elif args.arch == 'densenet':
        from models.cifar.densenet import DenseNet3
        net = DenseNet3(depth=info['depth'], num_classes=num_classes)
    # GoogLenet
    elif args.arch == 'googlenet':
        from models.cifar.googlenet import googlenet
        net = googlenet(num_classes=num_classes)
    # Not Supported
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net