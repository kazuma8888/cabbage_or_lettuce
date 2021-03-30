import cv2

class ComposeTransform():
    """
    複数のTransformをまとめあげる
    """
    def __init__(self, transforms=None):
        """
        Parameters
        --------------
        transforms: list
            transformのインスタンスをリストにして渡す
        """
        self.transforms = transforms

    def __call__(self, x):
        if self.transforms:
            for transform in self.transforms:
                x = transform(x)
        return x


class BaseTransform():
    """
    自作Transformの基底クラス
    """
    def __init__(self, debug=False):
        self.debug = debug
    
    def __call__(self):
        raise NotImplementedError()


class SimpleTransform(BaseTransform):
    """
    とりあえずのクラス
    よく使うものを入れておく
    扱う関数が増えてきたらテーマごとに分離する
    """
    def __init__(self, debug):
        super().__init__(debug)
        self.applied_transforms = [
            self.resize
        ]
    def __call__(self, x):
        if self.debug:
            print(x)
            for transform in self.applied_transforms:
                x = transform(x)
                print('-------------------')
                print(str(transform))
                print(x)
        else:
            for transform in self.applied_transforms:
                x = transform(x)
        return x

    
    def resize(self, x):
        return cv2.resize(x, (224, 224))
    
    def transpose(self, x):
        x = x.transpose(2, 0, 1)
        print(x.shape)
        return x