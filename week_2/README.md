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
