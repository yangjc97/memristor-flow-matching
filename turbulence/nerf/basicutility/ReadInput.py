import yaml


def read_yaml(filename: str) -> dict:
    """Reads a yaml file and returns a dictionary
    ----------
    Parameters
    * filename: str; the path to the yaml file


    Returns
    * dict; the dictionary containing the yaml entries
    """

    with open(filename, "r") as f:
        return yaml.safe_load(f)


class basic_input(object):
    
    defaults = {}

    def __init__(self, input_file: str):
        """
        Load a yaml file and store the entries as attributes of the class

        Parameters
        ----------
        * input_file: str; the path to the yaml file
        """
        self.yamlpath = input_file

        yaml_dict = read_yaml(input_file)

        for key in yaml_dict:
            setattr(self, key, yaml_dict[key])

        # set default values
        for key in self.__class__.defaults:
            if hasattr(self, key):
                pass
            else:
                setattr(self, key, self.__class__.defaults[key])
                print(f"Using default value for {key}: {self.__class__.defaults[key]}")

    @classmethod
    def update_defaults(cls, new_defaults:dict):
        cls.defaults.update(new_defaults)


class TrainParamReader(basic_input):
        
    def __init__(self, input_file: str):
        """
        Load a yaml file and store the entries as attributes of the class

        Parameters
        ----------
        * input_file: str; the path to the yaml file
        """
        super().__init__(input_file)

        self.lr: float
        self.epochs: int
        self.randomSeed: int
        self.savePath: str
        self.scheduler: dict
        self.optimizer: str
        self.batchSize: int
        self.dataPath: str


def createFn(fnName, pkgName):
    """
    Create nested function/class from a dictionary
    example:
        to create function/class `fo` from package which has the API: `foo(x, y, z)`
        where `x` is a function/class, which has the API: `fx(a, b)`,
            where `a` is a function/class, which has the API: `fa(alpha)`,
                where `alpha` is a regular parameter
        `y` and `z` are regular parameters (the parameters which don't require input)

        the yaml file should be like this:

        fo:
            name: foo
            kwargs:
                x:
                    name: fx
                    kwargs:
                        a:
                            name: fa
                            kwargs:
                                alpha: 1
                        b: 2
                y: 3
                z: 4

    ----------

    the above yaml file is equivalent to:
    >>> fo = abc.foo(
    >>>     x = fx( a = fa(alpha = 1), b = 2 ),
    >>>     y = 3,
    >>>     z = 4,
    >>> )
    """

    def getattr_multisource(pkgName, fnName):
        try: 
            iter(pkgName)
        except TypeError:
            pkgName = (pkgName,)

        counter = 0
        for i in pkgName:
            try:
                v = getattr(i, fnName)
                return v
            except AttributeError:
                pass
            counter += 1
        if counter == len(pkgName):
            raise AttributeError(
                f"Could not find {fnName} in {tuple(i.__name__ for i in pkgName)}"
            )

    if isinstance(fnName, dict):
        if "kwargs" in fnName:
            if len(fnName["kwargs"]) > 0:
                for k in fnName["kwargs"]:
                    if isinstance(fnName["kwargs"][k], dict):
                        fnName["kwargs"][k] = createFn(fnName["kwargs"][k], pkgName)
            v = getattr_multisource(pkgName, fnName["name"])(**fnName["kwargs"])
        else:
            v = getattr_multisource(pkgName, fnName["name"])
    else:
        v = getattr_multisource(pkgName, fnName)
    return v
