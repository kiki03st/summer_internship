# 2주차

## Pytorch 설치

Pytorch 설치를 위해 CUDA, cuDNN을 설치했다. 




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




