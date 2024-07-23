import numpy as np
from hashlib import sha256
import importlib
import json

def sizeOfShard(container, shard):
    '''
    Trả về số lượng ảnh khuôn mặt của một phân đoạn trước khi xuất hiện các yêu cầu loại bỏ ảnh khuôn mặt
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    return shards[shard].shape[0]

def realSizeOfShard(container, label, shard):
    '''
    Trả về số lượng ảnh khuôn mặt của một phân đoạn sau khi xuất hiện các yêu cầu loại bỏ ảnh khuôn mặt
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile:{}.npy'.format(container, label), allow_pickle=True)
    
    return shards[shard].shape[0] - requests[shard].shape[0]

def getShardHash(container, label, shard, until=None):
    '''
    Trả về giá trị băm của các vị trí của các ảnh khuôn mặt trong một phân đoạn nhỏ hơn biến until 
    (có thể hiểu là các vị trí của các ảnh khuôn mặt trên một lát cắt cụ thể) mà không có xuất hiện trong 
    các yêu cầu loại bỏ (được chia cắt bởi :). 
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile:{}.npy'.format(container, label), allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard]).astype('int')
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(container, label, shard, batch_size, dataset, offset=0, until=None):
    '''
    Trả về các lô chứa các ảnh khuôn mặt cần được huấn luyện, mỗi lô có kích thước là batch_size và
    không chứa các ảnh khuôn mặt bị yêu cầu loại bỏ.
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile:{}.npy'.format(container, label), allow_pickle=True)
 
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset

    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard]).astype('int')
        yield dataloader.load(indices)
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard]).astype('int')
        yield dataloader.load(indices)

def fetchTestBatch(dataset, batch_size):
    '''
    Trả về các lô chứa các ảnh khuôn mặt cần được kiểm thử, mỗi lô có kích thước là batch_size và
    không chứa các ảnh khuôn mặt bị yêu cầu loại bỏ.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), method='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(np.arange(limit, datasetfile['nb_test']), method='test')
        
def calcNumberRetrainedPoints(container, label, shard, batch_size, dataset, offset=0, until=None):
    '''
    Tính số lượng ảnh khuôn mặt được huấn luyện lại khi gặp ảnh khuôn mặt bị người dùng yêu cầu loại bỏ
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile:{}.npy'.format(container, label), allow_pickle=True)
    
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    numOfRetrainPoints = 0
    numPoints = 0
    isRetrain = False

    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard]) 
        
        if indices.shape[0] != shards[shard][limit-batch_size:limit].shape[0]: # If it is result of D / {d_u}, there are retrained points
            isRetrain = True

        numPoints += indices.shape[0]
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard]) 
        
        if indices.shape[0] != shards[shard][limit:until].shape[0]: # If it is result of D / {d_u}, there are retrained points
            isRetrain = True
                
        numPoints += indices.shape[0]

    if isRetrain == True:
        numOfRetrainPoints += numPoints

    return numOfRetrainPoints