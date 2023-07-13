# 2주차

## Pytorch 설치

Pytorch 설치를 위해 **CUDA**, **cuDNN**을 설치했다. 




**사용 환경**


GPU | CUDA | cuDNN
--- | --- | ---
1660 super | 10.0.130 | 10.0



처음에는 [Pytorch](<https://pytorch.org/get-started/locally/>) 사이트로 들어간 후, pip 패키지로 설치하기 위해 cmd에서 다음과 같은 명령어를 실행했다. 



    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu100



하지만 오류가 발생하면서 실패했다. 



```
Looking in indexes: https://download.pytorch.org/whl/cu100
Collecting torch
  Using cached https://download.pytorch.org/whl/cu100/torch-1.2.0-cp37-cp37m-win_amd64.whl (750.2 MB)
Collecting torchvision
  Using cached https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp37-cp37m-win_amd64.whl (1.0 MB)
Collecting torchaudio
  Using cached https://download.pytorch.org/whl/torchaudio-0.9.1-cp37-cp37m-win_amd64.whl (216 kB)
Requirement already satisfied: numpy in c:\users\ensung\appdata\local\packages\pythonsoftwarefoundation.python.3.7_qbz5n2kfra8p0\localcache\local-packages\python37\site-packages (from torch) (1.21.6)
INFO: pip is looking at multiple versions of torchvision to determine which version is compatible with other requirements. This could take a while.
Collecting torchvision
  Using cached https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl (1.5 MB)
  Using cached https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl (48 kB)
  Using cached https://download.pytorch.org/whl/torchvision-0.1.6-py3-none-any.whl (16 kB)
INFO: pip is looking at multiple versions of torchaudio to determine which version is compatible with other requirements. This could take a while.
Collecting torchaudio
  Using cached https://download.pytorch.org/whl/torchaudio-0.9.0-cp37-cp37m-win_amd64.whl (215 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.8.1-cp37-none-win_amd64.whl (109 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.8.0-cp37-none-win_amd64.whl (109 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.7.2-cp37-none-win_amd64.whl (103 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.7.1-cp37-none-win_amd64.whl (104 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.7.0-cp37-none-win_amd64.whl (103 kB)
  Using cached https://download.pytorch.org/whl/torchaudio-0.6.0-cp37-none-win_amd64.whl (85 kB)
INFO: pip is looking at multiple versions of torchaudio to determine which version is compatible with other requirements. This could take a while.
ERROR: Cannot install torch, torchaudio==0.6.0, torchaudio==0.7.0, torchaudio==0.7.1, torchaudio==0.7.2, torchaudio==0.8.0, torchaudio==0.8.1, torchaudio==0.9.0 and torchaudio==0.9.1 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested torch
    torchaudio 0.9.1 depends on torch==1.9.1
    The user requested torch
    torchaudio 0.9.0 depends on torch==1.9.0
    The user requested torch
    torchaudio 0.8.1 depends on torch==1.8.1
    The user requested torch
    torchaudio 0.8.0 depends on torch==1.8.0
    The user requested torch
    torchaudio 0.7.2 depends on torch==1.7.1
    The user requested torch
    torchaudio 0.7.1 depends on torch==1.7.1
    The user requested torch
    torchaudio 0.7.0 depends on torch==1.7.0
    The user requested torch
    torchaudio 0.6.0 depends on torch==1.6.0

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```



그래서 이번에는 **Anaconda**를 활용하여 Pytorch를 설치해보기로 했다. 

Anaconda 설치 후, Anaconda Prompt에서 다른 가상 환경 없이 base에서 명령어를 실행해보았다. 



    conda install pytorch torchvision torchaudio pytorch-cuda=10.0 -c pytorch -c nvidia



하지만 다음과 같은 오류가 발생했다. 



```
Collecting package metadata (current_repodata.json): done
Solving environment: unsuccessful initial attempt using frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: unsuccessful initial attempt using frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - pytorch-cuda=10.0

Current channels:

  - https://conda.anaconda.org/pytorch/win-64
  - https://conda.anaconda.org/pytorch/noarch
  - https://conda.anaconda.org/nvidia/win-64
  - https://conda.anaconda.org/nvidia/noarch
  - https://repo.anaconda.com/pkgs/main/win-64
  - https://repo.anaconda.com/pkgs/main/noarch
  - https://repo.anaconda.com/pkgs/r/win-64
  - https://repo.anaconda.com/pkgs/r/noarch
  - https://repo.anaconda.com/pkgs/msys2/win-64
  - https://repo.anaconda.com/pkgs/msys2/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.
```



그렇기에 이번에는 cmd에서 사용했던 pip 명령어를 Anaconda에서 실행해보았다. 

하지만 이번에도 오류가 발생하면서 실패했다. 



```
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
ERROR: No matching distribution found for torch
```



그래서 이번에는 Anaconda 내에서 가상 환경을 만들고, [이전 버전 Pytorch](<https://pytorch.org/get-started/previous-versions/>)를 다운받는 명령어로 다시 설치해보려 했다. 



    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch



하지만 이번에도 위의 conda 명령어 실행 시 떴던 오류와 비슷한 오류가 뜨면서 설치하지 못했다. 

혹시나 해서 pip로 설치하는 명령어를 사용해보았지만, 이 또한 오류가 발생하면서 Pytorch를 설치할 수 없었다. 




```
ERROR: Could not find a version that satisfies the requirement torch==1.2.0 (from versions: 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1)
ERROR: No matching distribution found for torch==1.2.0
```



결국 어쩔 수 없이 CUDA 10.0에 호환되는 Pytorch 대신, CUDA 10.1에 호환되는 것을 설치해보기로 했다. 

CUDA 10.1에 호환되는 Pytorch는 v1.4.0이었고, 해당하는 conda 명령어는 다음과 같았다. 



```anaconda
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
```



하지만 이것 또한 다음과 같은 오류가 발생하면서 설치할 수 없었다. 




```
Collecting package metadata (current_repodata.json): done
Solving environment: unsuccessful initial attempt using frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: unsuccessful initial attempt using frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - torchvision==0.5.0

Current channels:

  - https://conda.anaconda.org/pytorch/win-64
  - https://conda.anaconda.org/pytorch/noarch
  - https://repo.anaconda.com/pkgs/main/win-64
  - https://repo.anaconda.com/pkgs/main/noarch
  - https://repo.anaconda.com/pkgs/r/win-64
  - https://repo.anaconda.com/pkgs/r/noarch
  - https://repo.anaconda.com/pkgs/msys2/win-64
  - https://repo.anaconda.com/pkgs/msys2/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.
```



혹시나 해서 pip 명령어도 실행해보았지만 오류가 발생하면서 실행할 수 없었다. 



```
ERROR: Could not find a version that satisfies the requirement torch==1.4.0 (from versions: 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1)
ERROR: No matching distribution found for torch==1.4.0
```



그래서 버전을 올려 Pytorch 1.5.0을 설치해보기로 했다. 
아무래도 conda 명령어보다는 pip 명령어가 더 설치가 잘 되는 느낌이라 이번에는 pip 명령어 먼저 실행해보았다. 



    pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html



그랬더니 오류가 안뜨는가 싶더니 진짜 안뜨고 설치됐다. 왜일까...?

일단 설치된 김에 제대로 설치되었는지 확인하기 위해 vscode에서 파이썬 코드를 작성해서 실행해보았다. 



```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```



그랬더니 다음과 같은 출력 결과가 나왔다. 



```
True
NVIDIA GeForce GTX 1660 SUPER
```



그래픽 카드 이름도 제대로 출력되고, CUDA 사용 가능 여부도 True로 반환되었으므로 올바르게 Pytorch가 깔린 것이다. 

비록 권장 버전이 맞지는 않지만, 우여곡절 끝에 설치되어서 다행이다.  



문제가 생겼다. 

Tensorflow를 1.14.0으로 설치해야 하는 것을 알게 되어 설치했지만, 정작 기존에 깔려있던 Pytorch가 문제를 일으켰다. 

오류 내용은 다음과 같다. 



```
Traceback (most recent call last):
  File "C:\Users\ensung\anaconda3\envs\intern\lib\runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "C:\Users\ensung\anaconda3\envs\intern\lib\runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy\__main__.py", line 39, in <module>
    cli.main()
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy/..\debugpy\server\cli.py", line 430, in main
    run()
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy/..\debugpy\server\cli.py", line 284, in run_file
    runpy.run_path(target, run_name="__main__")
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 322, in run_path
    pkg_name=pkg_name, script_name=fname)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 136, in _run_module_code
    mod_name, mod_spec, pkg_name, script_name)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "C:\Users\ensung\Documents\test.py", line 2, in <module>
    import torch
  File "C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\torch\__init__.py", line 81, in <module>
    ctypes.CDLL(dll)
  File "C:\Users\ensung\anaconda3\envs\intern\lib\ctypes\__init__.py", line 364, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 127] 지정된 프로시저를 찾을 수 없습니다
```



인터넷에 검색해보니 관련 [게시글](<https://bnmy6581.tistory.com/47>)이 하나 나왔다. 

하지만 이미 설치했던 버전은 1.5.0이었고, 결국 다운그레이드를 계속 해봐야 할 것 같아서 1.2.0으로 다운그레이드를 시도해보았다. 

이번에는 이전에 썼던 1.2.0 다운로드 명령어와는 좀 다르게 변형해서 실행해보았다. 



    pip install torch==1.2.0+cu100 torchvision==0.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html



하지만 이 또한 오류가 발생했다. 



```
ERROR: Could not find a version that satisfies the requirement torch==1.2.0+cu100 (from versions: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.2.0+cpu, 1.2.0+cu92, 1.3.0, 1.3.0+cpu, 1.3.0+cu92, 1.3.1, 1.3.1+cpu, 1.3.1+cu92, 1.4.0, 1.4.0+cpu, 1.4.0+cu92, 1.5.0, 1.5.0+cpu, 1.5.0+cu101, 1.5.0+cu92, 1.5.1, 1.5.1+cpu, 1.5.1+cu101, 1.5.1+cu92, 1.6.0, 1.6.0+cpu, 1.6.0+cu101, 1.7.0, 1.7.0+cpu, 1.7.0+cu101, 1.7.0+cu110, 1.7.1, 1.7.1+cpu, 1.7.1+cu101, 1.7.1+cu110, 1.8.0, 1.8.0+cpu, 1.8.0+cu101, 1.8.0+cu111, 1.8.1, 1.8.1+cpu, 1.8.1+cu101, 1.8.1+cu102, 1.8.1+cu111, 1.9.0, 1.9.0+cpu, 1.9.0+cu102, 1.9.0+cu111, 1.9.1, 1.9.1+cpu, 1.9.1+cu102, 1.9.1+cu111, 1.10.0, 1.10.0+cpu, 1.10.0+cu102, 1.10.0+cu111, 1.10.0+cu113, 1.10.1, 1.10.1+cpu, 1.10.1+cu102, 1.10.1+cu111, 1.10.1+cu113, 1.10.2, 1.10.2+cpu, 1.10.2+cu102, 1.10.2+cu111, 1.10.2+cu113, 1.11.0, 1.11.0+cpu, 1.11.0+cu113, 1.11.0+cu115, 1.12.0, 1.12.0+cpu, 1.12.0+cu113, 1.12.0+cu116, 1.12.1, 1.12.1+cpu, 1.12.1+cu113, 1.12.1+cu116, 1.13.0, 1.13.0+cpu, 1.13.0+cu116, 1.13.0+cu117, 1.13.1, 1.13.1+cpu, 1.13.1+cu116, 1.13.1+cu117)
ERROR: No matching distribution found for torch==1.2.0+cu100
```



오류 코드를 자세히 읽어보니 torch 1.2.0+cu100과 torchvision 0.4.0+cu100 이외에 torch 1.2.0, torchvision 0.4.0이 링크 속에 존재했다.

그래서 명령어를 조금 수정한 뒤, 다시 실행해보았다. 



    pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html



그러니 놀랍게도 여태껏 깔리지 않았던 1.2.0 버전 Pytorch가 깔리기 시작했다. 

알고보니 torch의 버전은 1.2.0+cu92였고, torchvision의 버전은 0.4.0+cu92였다. 

원래 CUDA 9.2 버전에서 지원하도록 만든 것이 10.0에서도 호환이 되었던 것이라 설치 과정에서 어려움을 겪었던 것이 아닌가 생각했다. 



## TensorFlow

텐서플로우는 설치 과정이 비교적 쉬웠던 것 같다. 

CUDA 10.0과 cuDNN 7.4.2 버전에 맞게 tensorflow 2.0.0 버전을 pip 명령어를 통해 설치했다. 



    pip install tensorflow==2.0.0



그 후, tensorflow-gpu도 동일한 버전으로 설치했다. 



    pip install tensorflow-gpu==2.0.0



pytorch와는 달리 아무런 오류 없이 순조롭게 설치되었고, 올바르게 설치되었는지 확인하기 위해 파이썬 코드를 실행해보았다. 




```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```



하지만 오류 하나 없이 설치되었던 것과 달리, 실행 과정에서 오류가 발생했다. 



```
Traceback (most recent call last):
  File "C:\Users\ensung\anaconda3\envs\torch\lib\runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "C:\Users\ensung\anaconda3\envs\torch\lib\runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy\__main__.py", line 39, in <module>
    cli.main()
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy/..\debugpy\server\cli.py", line 430, in main
    run()
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher/../..\debugpy/..\debugpy\server\cli.py", line 284, in run_file
    runpy.run_path(target, run_name="__main__")
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 322, in run_path
    pkg_name=pkg_name, script_name=fname)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 136, in _run_module_code
    mod_name, mod_spec, pkg_name, script_name)
  File "c:\Users\ensung\.vscode\extensions\ms-python.python-2023.12.0\pythonFiles\lib\python\debugpy\_vendored\pydevd\_pydevd_bundle\pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "C:\Users\ensung\Documents\test.py", line 1, in <module>
    import tensorflow
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow\__init__.py", line 98, in <module>
    from tensorflow_core import *
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\__init__.py", line 40, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  File "<frozen importlib._bootstrap>", line 959, in _find_and_load_unlocked
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow\__init__.py", line 50, in __getattr__
    module = self._load()
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow\__init__.py", line 44, in _load
    module = _importlib.import_module(self.__name__)
  File "C:\Users\ensung\anaconda3\envs\torch\lib\importlib\__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\python\__init__.py", line 52, in <module>
    from tensorflow.core.framework.graph_pb2 import *
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\graph_pb2.py", line 16, in <module>
    from tensorflow.core.framework import node_def_pb2 as tensorflow_dot_core_dot_framework_dot_node__def__pb2
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\node_def_pb2.py", line 16, in <module>
    from tensorflow.core.framework import attr_value_pb2 as tensorflow_dot_core_dot_framework_dot_attr__value__pb2
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\attr_value_pb2.py", line 16, in <module>
    from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\tensor_pb2.py", line 16, in <module>
    from tensorflow.core.framework import resource_handle_pb2 as tensorflow_dot_core_dot_framework_dot_resource__handle__pb2
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\resource_handle_pb2.py", line 16, in <module>
    from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\tensorflow_core\core\framework\tensor_shape_pb2.py", line 42, in <module>
    serialized_options=None, file=DESCRIPTOR),
  File "C:\Users\ensung\anaconda3\envs\torch\lib\site-packages\google\protobuf\descriptor.py", line 561, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```



오류 코드를 읽어보니 protobuf의 버전을 다운그레이드 하는 방법으로 해결할 수 있다고 한다. 그래서 다음과 같은 명령어를 실행했다. 



    pip install protobuf==3.20.*



이후, 파이썬 코드를 재실행해보니 다음과 같은 결과가 나오면서 정상적으로 작동했다. 



```python
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 6033115836984333028
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 4974444544
locality {
  bus_id: 1
  links {
  }
}
incarnation: 6002390634539786132
physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1660 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5"
]
```



정석적으로 따라가니 쉽게 해결할 수 있었던 것 같다. 



문제가 생겼다. 

Tensorflow를 1.14.0 버전으로 깔아야 했었다. 

YOLO v3에 맞추기 위해 기존에 깔았던 Tensorflow 2.0.0 버전을 삭제하고 다음 명령어를 통해 재설치했다. 



```
pip install tensorflow==1.14.0
pip install tensorflow-gpu==1.14.0
```



이후 파이썬 코드를 실행시켰을 때, 오류가 발생했지만 Pytorch의 버전 다운그레이드를 통해 문제를 해결했다. 

문제 해결 후에 파이썬 코드를 돌리니 Tensorflow v2.0.0, Pytorch v1.5.0일 때처럼 올바르게 내용이 출력되었다. 

하지만 조금 거슬리는 정도의 문제점이 발생했다. 

실행 자체에는 문제가 되지 않지만, 경고문 정도로 터미널에 문구가 계속 뜬다. 



```
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorflow\python\framework\dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
C:\Users\ensung\anaconda3\envs\intern\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
2023-07-13 15:59:42.653927: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2023-07-13 15:59:42.660243: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library nvcuda.dll
2023-07-13 15:59:42.713735: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties: 
name: NVIDIA GeForce GTX 1660 SUPER major: 7 minor: 5 memoryClockRate(GHz): 1.815
pciBusID: 0000:01:00.0
2023-07-13 15:59:42.719670: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2023-07-13 15:59:42.722863: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0
2023-07-13 15:59:43.064935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2023-07-13 15:59:43.067493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0
2023-07-13 15:59:43.069052: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N
2023-07-13 15:59:43.070826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/device:GPU:0 with 4752 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1660 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5)
```



아무래도 numpy에서 문제가 생긴 것 같아 numpy 버전을 1.21.6에서 1.16.4로 다운그레이드 시켜보기로 했다. 



    pip install numpy==1.16.4



명령어를 실행하여 버전을 다운그레이드하니 거슬리던 경고문구들도 없어지게 되었다. 
