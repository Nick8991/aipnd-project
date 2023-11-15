from torch import nn

def vgg1():
    classifier = nn.Sequential(nn.Linear(25088,1568),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(1568,392),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(392,102),
                    nn.LogSoftmax(dim=1))
    return classifier

def vgg2():
    classifier = nn.Sequential(nn.Linear(25088,3136),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(3136,1568),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(1568,392),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(392,102),
                            nn.LogSoftmax(dim=1))
    return classifier

def alexnet1():
    classifier = nn.Sequential(
        nn.Linear(9216, 4096),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )
    return classifier

def alexnet2():
    classifier = nn.Sequential(
        nn.Linear(9216, 4096),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 1024), 
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 102),   
        nn.LogSoftmax(dim=1)    
    )
    return classifier

def alexnet3():
    classifier = nn.Sequential(
        nn.Linear(9216, 4096),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 2048),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(2048, 1024),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 512),  
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 102),    
        nn.LogSoftmax(dim=1)
    )
    return classifier