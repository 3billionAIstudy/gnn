# GNN - 그래프 자료구조와 geometric deep learning

생성일: 2023년 1월 30일 오전 11:03
태그: deep learning, gnn

### Graph란?

- **vertex**와 **edge**로 이루어진 **Non-Euclidean** 자료구조이며 **요소 간의 연결관계**를 나타낼 수 있다.
- social network, physical system, protein-protein interaction, knowledge graph등이 그래프로 표현될 수 있다.

![그래프 자료구조의 예시](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled.png)

그래프 자료구조의 예시

- 그래프 자료구조의 표현법은 아래와 같다.

![그래프의 표현법](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%201.png)

그래프의 표현법

- 여기서 우리가 보는 그래프의 구조와 실제 컴퓨터가 처리하기 쉬운 그래프의 구조는 조금 다르다. 실제 그래프를 자료구조 표현할 때 **흔히 사용 되는 방식**은 adjacency matrix를 이용하는 방식이다.
    - **adjacency matrix**는 노드 간의 연결여부를 0과 1로 인코딩한 2차원 배열 형태의 자료 구조이다.
    - 여기서 부가적으로 각 노드 혹은 엣지가 가지는 특성을 표현하고 하는 경우 feature matrix를 추가적으로 사용한다.
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%202.png)
    
     
    

### 그래프의 종류

- 그래프를 **구분하는 축은 크게 아래 세 가지**이다. 이외에도 그래프의 순환여부에 따라 Cyclic/Acyclic으로도 구분하다.
    
    1) **Directed/Undirected**: 엣지의 방향, 즉 출발노드와 도착노드가 정해진 경우는 Directed graph, 방향이 없는 경우는 Undirected graph라고 한다.
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%203.png)
    
    2) **Homogenous/Heterogeneous**: 노드와 엣지의 종류가 모두 같은 경우는 Homogenous graph  노드와 엣지에 종류가 있는 경우는 Heterogeneous graph라고 한다.
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%204.png)
    
    3) **Static/Dynamic**: 시간에 따라 그래프가 변하는 경우 Dynamic graph, 그렇지 않은 경우는 Static graph라고 한다.
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%205.png)
    

### Graph Neural Network

- 최근에 Neural Network이 이미지나 음성, 자연어 같은 도메인에서 크게 성공하면서 그래프나 manifold와 같은 **Non-Euclidean data**에 **확장하여** 적용하려는 시도가 많이 있었다. (이 분야를 흔히 **geometric deep learning**이라고 지칭 한다.)
- 허나 기존의 **convolution**과 같은 Neural Network의 **핵심 연산**을 **Non-Euclidean data**에 바로 적용할 수 없다는 문제가 있었다.

![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%206.png)

### Non-Euclidean data와 geometric deep learning

- ‘Non-Euclidean data’는 prof. Bronstein에 의해서 처음 소개된 개념으로 geometric deep learning을 설명하며 나온 개념이다.
- **Non-Euclidean data**란 넓게 말해서 데이터의 도메인 상의 두 좌표점에 대해서 **유클리드 거리를 이용할 수 없는 모든 데이터**를 말한다.
    - 그리고 **그래프 데이터는 이러한 Non-Euclidean data 중 하나**이다.
    - 반면에 이미지와 같은 euclidean data의 경우 예로 보면 이미지는 좌표 상의 강도 혹은 색깔을 표현하는 형태로 볼 수 있다.
    - ****(이미지)신호를 ****f: (x, y) → R 형태의 함수로 표현 가능하고 두 데이터 포인트, A=(a, b), C=(c, d) 간의 거리는 $d(A, C) = \sqrt{(a-c)^2 + (b-d)^2}$로 표현이 가능하다. 이렇듯 이미지는 명백히 Euclidean data이다.
- 이러한 성질이 문제가 되는 이유는 다음과 같다. 우리가 딥러닝 모델을 만들며 흔히 적용하는 **convolution 연산**은 **euclidean domain에서 정의**되어 있고 따라서 Non-Euclidean data에 대해서 **바로 적용될 수 없다.**
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%207.png)
    

### Convolution mechanism을 적용하는 이유

- 우리가 gnn으로 풀고자 하는 문제는 노드가 가지고 있는 정보(노드의 종류와 같은)와 노드 간의 연결정보 (즉  전체 그래프에서 topology 정보)를 dense한 벡터로 밀집시키는문제이다.
    - 만들어진 밀집벡터를 이용해 회귀문제나 분류문제, 디코더를 연결하여 생성모델을 만들 수 있게 된다.

![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%208.png)

- 이러한 부분에서 **CNN을 이용해야된다는 발상**이 시작되었다. CNN의 특징 중 1)**local connection**과 2) **shared weight**와 같은 특징등 덕분에 위와 같은 문제를 잘 풀 수 있다.
    - 해당 성질들에 의해서 translation equivariant의 성질을 가지므로 그래프를 임베딩 하는 데에 도움이 된다.

![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%209.png)

![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2010.png)

- 허나 convolution 개념을 Non-Euclidean 데이터에 적용하기 위해서는 새로운 정의가 필요하다.
    - 이를 위해서 **Spectral approaches**라는 방식이 주로 사용 되는 데, 신호 처리에서 신호를 주파수 도메인에서 해석하는 방법을 그래프에 적용하여 convolution을 수행하는 방법이다.
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2011.png)
    

### GNN의 구성요소

![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2012.png)

- 위와 같은 형태가 가장 보편적인 Graph neural network의 구조이다.
- 위의 구조에서 GNN Layer를 보면 1) conv/recurrent operator 2) sampling operator 3) pooling operator 이 세 가지 구성요소를 가진다.
    
    **1) conv/recurrent operator(propagation module)**
    
    - GNN의 경우 임베딩을 만들기 위해 layer를 거치며 인접 노드들의 정보를 취합하여 target node에 전달하는 방식으로 작동한다.
    - 아래 그림에서 보면 target node로부터 출발하여 target node로 들어오는 노드들의 정보를 취합해 나가는 것을 알 수 있다.
    - 해당 aggregation 연산을 수행하기 위해서 convolution이나 recurrent unit을 활용한다.(주로 convolution 연산을 더 자주 이용한다.)
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2013.png)
    
    **2) sampling operator**
    
    - 위의 propagation 과정에서 target 노드에서 인접노드 전체를 보는 것이 아니라 **일부 노드 만을 보는 것**을 알 수 있다. 이러한 샘플링을 수행하는 모듈이 sampling module이다.
    - 이러한 샘플링을 하는 이유는 **노드의 수가 증가**할 수록 **propagation을 위한 연산량이 기하급수적으로 증가**하기 때문이다.
    - sampling module의 종류로는 크게 node sampling, layer sampling, subgraph sampling등이 있다.
    
    **3) pooling operator** 
    
    - 위의 propagation을 통해 모인 정보를 실제 row dimension의 밀집벡터로 projection 시키는 일은 pooling module이 하게 된다.
    - GNN에서 사용되는 pooling은  크게 1) direct pooling과 2) hierarchical pooling  두 종류로 볼 수 있다.
        
        1) direct pooling
        
        - 흔히 ‘**Readout**’이라고 불리며 convolution filter들에 의해 생긴 각각의
        
        ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2014.png)
        
    
    ![Untitled](GNN%20-%20%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%84%91%E1%85%B3%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AA%20geometric%20deep%20learning%20dfd497c3464c4d459c7923a970f0ca9f/Untitled%2015.png)
    

### 정리

- 그래프 자료구조와 GNN에 대한 소개를 간략하게 작성해보았다. 글의 내용을 정리해보면 아래와 같다.

- 그래프는 node와 edge로 구성된 자료구조이며 Non-Euclidean data이다.
- 분자구조, protein-protein interaction network, 지식 그래프등에 활용된다.
- 주로 adjacency matrix와 feature matrix를 이용하여 표현한다.

- GNN은 크게 propagation module, sampling module, pooling module 세 가지로 구성된다.
    - 이중 propagation module에는 convolution 연산이 이용되는 데, Non-Euclidean data에 적용하기 위해 새롭게 정의하여 사용한다.
    - propagation의 연산량은 노드 수에 따라 기하급수적으로 증가하면 때문에 sampling module이 필요하다.
    - pooling module을 통해 차원축소가 가능하다.

- 다음 글에서는 GNN을 구성하는 요소 중 propagation module에 대해서 자세히 알아볼 것이다.

---

### Reference

- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. http://arxiv.org/abs/2104.13478
- Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2020). Graph neural networks: A review of methods and applications. *AI Open*, *1*(December 2020), 57–81. https://doi.org/10.1016/j.aiopen.2021.01.001
- Monti, F., Boscaini, D., Masci, J., Rodolà, E., Svoboda, J., & Bronstein, M. M. (2017). Geometric deep learning on graphs and manifolds using mixture model CNNs. *Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017*, *2017*-*January*, 5425–5434. https://doi.org/10.1109/CVPR.2017.576