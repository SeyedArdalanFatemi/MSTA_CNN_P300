"""MSTA-CNN: offline P300 decoding on BCI Competition III, dataset II.
Single-file implementation. See README.md for protocols and reproducibility limits.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, os, platform, sys, time, traceback, uuid, gc, zipfile
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import buttord, butter, sosfiltfilt, firwin, resample_poly
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)
SEED, EPOCHS, BATCH_SIZE = 42, 200, 32
FS_IN, FS_OUT, CHANNELS = 240, 64, 64
N_RAW, N_OUT, UP, DOWN, CONTEXT = 192, 52, 4, 15, 150
PAD_MAIN = 12 * FS_IN
ORDER, WN = buttord([.5,20],[.1,40],1.5,20,fs=FS_IN)
SOS = butter(ORDER,WN,btype='bandpass',fs=FS_IN,output='sos')
RESAMPLE_FIR = firwin(301,1.0/DOWN,window=('kaiser',5.0))
MATRIX='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'
VARIANT='none'
EXPECTED_PARAMS={'none':2001,'channel':2129,'temporal':3057,'full':3185}
SEEDS=(42,123,2026)
def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def validate_partition(X,y,meta,expected_chars):
    require(X.dtype==np.float32 and X.shape==(expected_chars*180,64,52,1),
            f'Unexpected tensor dtype/shape: {X.dtype}, {X.shape}')
    require(y.shape==(len(X),) and np.isin(y,[0,1]).all(),'Invalid binary labels')
    require(meta.shape==(len(X),4) and np.issubdtype(meta.dtype,np.integer),'Invalid metadata')
    require(np.isfinite(X).all(),'Non-finite EEG values')
    groups=np.unique(meta[:,0])
    require(len(groups)==expected_chars,'Unexpected number of characters')
    for g in groups:
        ix=np.flatnonzero(meta[:,0]==g)
        require(len(ix)==180,'Incomplete character')
        target=None
        for seq in range(15):
            rows=ix[meta[ix,1]==seq]
            codes=meta[rows,2]
            require(np.array_equal(np.sort(codes),np.arange(1,13)),'Incomplete/duplicate sequence')
            targets=tuple(sorted(codes[y[rows]==1].tolist()))
            require(len(targets)==2 and targets[0]<=6<targets[1],'Invalid target row/column')
            if target is None: target=targets
            require(target==targets,'Target identity changes inside character')
    return groups

def build_model(tf):
    L=tf.keras.layers
    x0=L.Input((64,52,1),name='eeg')
    x=L.Conv2D(8,(1,32),padding='same',use_bias=False,name='temporal_conv')(x0)
    x=L.BatchNormalization(name='bn1')(x)
    x=L.DepthwiseConv2D((64,1),depth_multiplier=2,use_bias=False,
        depthwise_constraint=tf.keras.constraints.max_norm(1.),name='spatial_conv')(x)
    x=L.BatchNormalization(name='bn2')(x)
    x=L.Activation('elu')(x)
    x=L.AveragePooling2D((1,4))(x)
    x=L.Dropout(.5)(x)
    x=L.SeparableConv2D(16,(1,16),padding='same',use_bias=False,name='separable_conv')(x)
    x=L.BatchNormalization(name='bn3')(x)
    x=L.Activation('elu')(x)
    x=L.AveragePooling2D((1,4))(x)
    x=L.Dropout(.5)(x)
    require(VARIANT in EXPECTED_PARAMS,'Unknown variant')
    if VARIANT in ('channel','full'):
        se=L.GlobalAveragePooling2D()(x)
        se=L.Dense(4,activation='relu',use_bias=False)(se)
        se=L.Dense(16,activation='sigmoid',use_bias=False)(se)
        se=L.Reshape((1,1,16),name='feature_attention')(se)
    if VARIANT in ('temporal','full'):
        local=L.Reshape((3,16))(x)
        local=L.Conv1D(16,3,padding='same',activation='relu')(local)
        local=L.Conv1D(16,1,padding='same',activation='sigmoid')(local)
        local=L.Reshape((1,3,16),name='temporal_attention')(local)
    if VARIANT!='none':
        refined=x
        if VARIANT in ('channel','full'): refined=L.Multiply()([refined,se])
        if VARIANT in ('temporal','full'): refined=L.Multiply()([refined,local])
        x=L.Add()([x,refined])
    x=L.Flatten(name='features')(x)
    out=L.Dense(1,activation='sigmoid',kernel_constraint=tf.keras.constraints.max_norm(.25))(x)
    model=tf.keras.Model(x0,out,name='MSTA_CNN_pilot')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss='binary_crossentropy',metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                  tf.keras.metrics.AUC(name='auc')])
    require(model.count_params()==EXPECTED_PARAMS[VARIANT],'Unexpected model parameter count')
    return model

def predict_batches(model,X):
    p=np.concatenate([np.asarray(model(X[i:i+128],training=False)).reshape(-1)
                      for i in range(0,len(X),128)])
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    return p

def binary_metrics(y,p):
    hard=(p>=.5).astype(int)
    return {'threshold':.5,'roc_auc':float(roc_auc_score(y,p)),
        'average_precision':float(average_precision_score(y,p)),
        'accuracy':float(accuracy_score(y,hard)),
        'target_precision':float(precision_score(y,hard,zero_division=0)),
        'target_recall':float(recall_score(y,hard,zero_division=0)),
        'target_f1':float(f1_score(y,hard,zero_division=0)),
        'unweighted_bce':float(log_loss(y,p,labels=[0,1])),
        'confusion_matrix_rows_true_cols_pred_0_1':confusion_matrix(y,hard,labels=[0,1]).tolist(),
        'always_nontarget_accuracy':float(np.mean(y==0))}

def character_metrics(y,p,meta):
    rows=[]; groups=np.unique(meta[:,0])
    for g in groups:
        ix=np.flatnonzero(meta[:,0]==g)
        scores=np.full((15,12),np.nan)
        for i in ix: scores[meta[i,1],meta[i,2]-1]=p[i]
        require(np.isfinite(scores).all(),'Missing character scores')
        targets=np.unique(meta[ix[y[ix]==1],2])
        require(len(targets)==2 and targets[0]<=6<targets[1],'Invalid character truth')
        truth=(targets[1]-7)*6+targets[0]-1
        sums=np.cumsum(scores,axis=0)
        for n in range(1,16):
            a=sums[n-1]; pred=int(np.argmax(a[6:])*6+np.argmax(a[:6]))
            rows.append({'character_id':int(g),'sequences':n,'true_character':MATRIX[truth],
                'predicted_character':MATRIX[pred],'correct':int(pred==truth)})
    curve=[{'sequences':n,'correct':sum(r['correct'] for r in rows if r['sequences']==n),
        'characters':len(groups),'accuracy':float(np.mean([r['correct'] for r in rows if r['sequences']==n]))}
        for n in range(1,16)]
    return curve,rows

def rising(values):
    active = np.asarray(values).reshape(-1) > 0
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])

def check_events(code, flashing, types, char_id, n_samples):
    prefix = f'Character {char_id} (zero-based): '
    require(code.shape == (n_samples,), prefix + 'StimulusCode shape mismatch')
    require(np.isfinite(code).all(), prefix + 'non-finite stimulus codes')
    require(np.isin(code, np.arange(13)).all(), prefix + 'invalid stimulus code')
    starts = rising(code)
    require(len(starts) == 180, prefix + f'expected 180 onsets; found {len(starts)}')
    # Codes must stay constant throughout each non-zero pulse.
    active_pairs = (code[1:] > 0) & (code[:-1] > 0)
    require(not np.any(active_pairs & (code[1:] != code[:-1])),
            prefix + 'code changed without a zero interval')
    if flashing is not None:
        require(flashing.shape == code.shape, prefix + 'Flashing shape mismatch')
        require(np.isin(flashing, [0, 1]).all(), prefix + 'Flashing is not binary')
        require(np.array_equal(starts, rising(flashing)),
                prefix + 'Flashing and StimulusCode onsets disagree')
    require(types.shape == code.shape, prefix + 'StimulusType shape mismatch')
    require(np.isin(types, [0, 1]).all(), prefix + 'StimulusType is not binary')
    event_codes = code[starts].astype(np.int16)
    labels = types[starts].astype(np.int8)
    target_codes = None
    for seq in range(15):
        sl = slice(seq * 12, (seq + 1) * 12)
        c, y = event_codes[sl], labels[sl]
        require(np.array_equal(np.sort(c), np.arange(1, 13)),
                prefix + f'sequence {seq}: codes 1..12 must each occur once')
        targets = tuple(sorted(c[y == 1].tolist()))
        require(len(targets) == 2 and targets[0] <= 6 < targets[1],
                prefix + f'sequence {seq}: expected one target column and one target row')
        if target_codes is None:
            target_codes = targets
        require(targets == target_codes, prefix + f'sequence {seq}: target identity changed')
    incomplete = starts[starts + N_RAW > n_samples]
    require(len(incomplete) == 0,
            prefix + f'{len(incomplete)} incomplete epochs; nothing silently discarded')
    return starts, event_codes, labels

def filtered(car, padlen=PAD_MAIN, padtype='odd'):
    require(len(car) > padlen + 1, f'Record too short for {padlen / FS_IN:g}-second padding')
    result = sosfiltfilt(SOS, car, axis=0, padtype=padtype, padlen=padlen)
    require(np.isfinite(result).all(), 'Non-finite filter output')
    return result

def epoch_at(processed, start, pad_mode='reflect'):
    left, right = int(start) - CONTEXT, int(start) + N_RAW + CONTEXT
    chunk = processed[max(0, left):min(len(processed), right)]
    pads = max(0, -left), max(0, right - len(processed))
    if any(pads):
        chunk = np.pad(chunk, ((pads[0], pads[1]), (0, 0)), mode=pad_mode)
    result = resample_poly(chunk, UP, DOWN, axis=0, window=RESAMPLE_FIR)
    offset = CONTEXT * UP // DOWN
    epoch = result[offset:offset + N_OUT].T
    require(epoch.shape == (CHANNELS, N_OUT), 'Unexpected epoch shape')
    return epoch, bool(any(pads))

def make_splits(nchars=85):
    require(nchars==85,'This protocol is fixed at 85 characters')
    outer=np.array_split(np.random.default_rng(42).permutation(nchars),5)
    splits=[]
    for fold,heldout in enumerate(outer,1):
        heldout=np.sort(heldout)
        development=np.setdiff1d(np.arange(nchars),heldout)
        inner=np.sort(np.random.default_rng(1000+fold).permutation(development)[:7])
        train=np.setdiff1d(development,inner)
        require(len(train)==61 and len(inner)==7 and len(heldout)==17,'Wrong split sizes')
        require(not set(train)&set(inner) and not set(train)&set(heldout) and not set(inner)&set(heldout),
                'Character overlap')
        splits.append({'fold':fold,'train_characters':train.tolist(),'inner_characters':inner.tolist(),
                       'outer_characters':heldout.tolist(),'inner_split_seed':1000+fold})
    require(np.array_equal(np.sort(np.concatenate(outer)),np.arange(nchars)), 'Outer coverage failed')
    return splits

def load_raw(path,folder):
    folder.mkdir(parents=True,exist_ok=True)
    source_hash=digest(path)
    manifest=folder/'raw_cache.json'
    if manifest.exists():
        info=json.loads(manifest.read_text(encoding='utf-8'))
        require(info['source_sha256']==source_hash,'Raw input changed; create a new run')
        for name,h in info['cache_sha256'].items():
            require((folder/name).is_file() and digest(folder/name)==h,'Raw cache missing or modified; create a new run')
        X=np.load(folder/'epochs_unnormalized.npy',mmap_mode='r',allow_pickle=False)
        with np.load(folder/'epoch_metadata.npz',allow_pickle=False) as z: y,meta=z['labels'],z['metadata']
        return X,y,meta,info
    mat=loadmat(path)
    for k in ['Signal','StimulusCode','StimulusType']: require(k in mat,'Missing '+k)
    signal,code,types=mat['Signal'],mat['StimulusCode'],mat['StimulusType']
    require(signal.ndim==3 and signal.shape[0]==85 and signal.shape[2]==64,'Unexpected Signal dimensions')
    nc,nt,_=signal.shape
    require(code.shape==types.shape==(nc,nt),'Unexpected event dimensions')
    flash=mat.get('Flashing')
    if flash is not None: require(flash.shape==code.shape,'Unexpected Flashing dimensions')
    X=np.empty((85*180,64,52,1),dtype=np.float32)
    y=np.empty(85*180,dtype=np.int8); meta=np.empty((85*180,4),dtype=np.int32)
    for char in range(nc):
        raw=np.asarray(signal[char],dtype=np.float64)
        require(np.isfinite(raw).all(),'Non-finite EEG')
        starts,codes,labels=check_events(code[char],None if flash is None else flash[char],types[char],char,nt)
        car=raw-raw.mean(axis=1,keepdims=True)
        proc=filtered(car)
        for event,start in enumerate(starts):
            i=char*180+event; ep,_=epoch_at(proc,start)
            X[i,:,:,0]=ep; y[i]=labels[event]; meta[i]=[char,event//12,codes[event],start]
        if (char+1)%10==0 or char==84: print(f'  Preprocessed {char+1}/85 characters',flush=True)
    validate_partition(X,y,meta,85)
    np.save(folder/'epochs_unnormalized.npy',X)
    np.savez(folder/'epoch_metadata.npz',labels=y,metadata=meta,sos=SOS,resampling_fir=RESAMPLE_FIR)
    info={'source_file':path.name,'source_sha256':source_hash,'characters':85,'epochs':len(y),
        'target_epochs':int(y.sum()),'discarded_epochs':0,'normalization_applied':False,
        'cache_sha256':{n:digest(folder/n) for n in ['epochs_unnormalized.npy','epoch_metadata.npz']}}
    write(manifest,info)
    return X,y,meta,info

def prepare_fold(X,meta,split):
    indices={name:np.flatnonzero(np.isin(meta[:,0],split[name+'_characters'])) for name in ['train','inner','outer']}
    train=np.array(X[indices['train']],dtype=np.float32,copy=True)
    mean=float(train.mean(dtype=np.float64)); std=float(train.std(dtype=np.float64))
    require(np.isfinite(std) and std>1e-12,'Invalid training scale')
    scale=std+1e-6
    arrays={'train':train}
    for name in ['inner','outer']: arrays[name]=np.array(X[indices[name]],dtype=np.float32,copy=True)
    for a in arrays.values():
        a-=mean; a/=scale; require(np.isfinite(a).all(),'Non-finite normalization output')
    return arrays,indices,{'mean':mean,'std':std,'epsilon':1e-6,'scale':scale,'fit_on':'61 training characters only'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def replace(src,dst):
    for n in range(21):
        try: os.replace(src,dst); return
        except PermissionError:
            if n==20: raise
            time.sleep(min(.1*(n+1),1.))

def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        replace(temp,path)
    finally: temp.unlink(missing_ok=True)

def save_model(model,path):
    temp=path.with_name('saving_'+uuid.uuid4().hex+'.keras')
    try: model.save(temp); replace(temp,path)
    finally: temp.unlink(missing_ok=True)

def score(m,p,y,meta,idx):
    require(p.shape==(len(idx),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    curve,rows=m.character_metrics(y[idx],p,meta[idx])
    return {'inner_binary':m.binary_metrics(y[idx],p),'inner_character_curve':curve,
            'inner_mean_accuracy_repetitions_1_to_15':float(np.mean([x['accuracy'] for x in curve])),
            'inner_accuracy_15':curve[-1]['accuracy']},rows

def store_predictions(dest,p,y,meta,idx):
    tmp=dest/('predictions_'+uuid.uuid4().hex+'.npz')
    np.savez_compressed(tmp,probability=p,truth=y[idx],metadata=meta[idx],indices=idx)
    replace(tmp,dest/'inner_predictions.npz')

def check_dropout(tf,model,rate):
    layers=[layer for layer in model.layers if isinstance(layer,tf.keras.layers.Dropout)]
    require(len(layers)==2 and all(float(l.rate)==rate and l.get_config()['rate']==rate for l in layers),'Unexpected Dropout configuration')

def set_dropout(tf,model,rate):
    require(rate==.25,'Only predetermined rate 0.25 is allowed')
    check_dropout(tf,model,.5)
    for layer in model.layers:
        if isinstance(layer,tf.keras.layers.Dropout):layer.rate=rate
    check_dropout(tf,model,rate)

def checkpoint_rates(path,rate):
    with zipfile.ZipFile(path) as z:cfg=json.loads(z.read('config.json'))
    layers=cfg['config']['layers'];rates=[x['config']['rate'] for x in layers if x['class_name']=='Dropout']
    require(rates==[rate,rate],'Saved checkpoint dropout rates differ')

def train_cv_fold(tf,m,arrays,idx,y,meta,variant,dest,stats,seed,dropout):
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    m.VARIANT=variant;m.SEED=seed;model=m.build_model(tf)
    if dropout==.25:set_dropout(tf,model,.25)
    options=tf.data.Options();options.experimental_deterministic=True
    options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=1
    weights=np.ones(len(idx['train']),dtype=np.float32)
    train_ds=tf.data.Dataset.from_tensor_slices((arrays['train'],y[idx['train']].astype(np.float32),weights))
    train_ds=train_ds.shuffle(len(weights),seed=seed,reshuffle_each_iteration=True).batch(32).with_options(options).prefetch(1)
    inner=tf.data.Dataset.from_tensor_slices((arrays['inner'],y[idx['inner']].astype(np.float32))).batch(32).with_options(options).prefetch(1)
    probe=arrays['inner'][:32]
    class Audit(tf.keras.callbacks.Callback):
        def __init__(self):super().__init__();self.best=float('inf');self.best_epoch=0;self.probe=None
        def on_epoch_begin(self,epoch,logs=None):self.lr=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        def on_train_batch_end(self,batch,logs=None):
            require(logs is not None and all(np.isfinite(float(v)) for v in logs.values()),'Non-finite training metric')
        def on_epoch_end(self,epoch,logs=None):
            require(logs is not None and all(np.isfinite(float(v)) for v in logs.values()),'Non-finite epoch metric')
            logs['learning_rate_used']=self.lr
            if logs['val_loss']<self.best:
                self.best=float(logs['val_loss']);self.best_epoch=epoch+1
                self.probe=np.asarray(self.model(probe,training=False));save_model(self.model,dest/'best.keras')
            write(dest/'progress.json',{'epochs_completed':epoch+1,'best_epoch':self.best_epoch,'best_inner_loss':self.best,'metrics':{k:float(v) for k,v in logs.items()}})
    audit=Audit()
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',factor=.5,patience=5,min_delta=.0001,min_lr=0.),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=20,min_delta=0.,restore_best_weights=False),audit]
    start=time.perf_counter();history=model.fit(train_ds,validation_data=inner,epochs=200,callbacks=callbacks,verbose=2)
    seconds=time.perf_counter()-start;loaded=tf.keras.models.load_model(dest/'best.keras',safe_mode=True)
    check_dropout(tf,loaded,dropout)
    diff=float(np.max(np.abs(np.asarray(loaded(probe,training=False))-audit.probe)))
    require(np.allclose(np.asarray(loaded(probe,training=False)),audit.probe,rtol=1e-6,atol=1e-7),'Reload verification failed')
    p=m.predict_batches(loaded,arrays['inner']);metrics,rows=score(m,p,y,meta,idx['inner'])
    result={**metrics,'training_seconds':seconds,'best_epoch':audit.best_epoch,'best_inner_loss':audit.best,
            'epochs_completed':len(history.history['loss']),'history':{k:[float(v) for v in vv] for k,vv in history.history.items()},
            'reload_max_probability_difference':diff,'normalization':stats,'checkpoint_sha256':m.digest(dest/'best.keras')}
    store_predictions(dest,p,y,meta,idx['inner']);write(dest/'inner_characters.json',rows)
    del model,loaded,train_ds,inner;tf.keras.backend.clear_session();gc.collect()
    return result

def verify_completed(tf,m,folder,task,protocol_sha,X):
    r=read(folder/'final_report.json')
    m.require(r['status']=='COMPLETED' and r['task']==task and r['protocol_sha256']==protocol_sha,'Completed final task mismatch')
    m.require(m.digest(folder/'final.keras')==r['checkpoint_sha256'],'Final checkpoint changed')
    m.require(m.digest(folder/'train_probe.npz')==r['probe_sha256'],'Training probe changed')
    tf.keras.backend.clear_session();model=tf.keras.models.load_model(folder/'final.keras',safe_mode=True)
    m.require(model.count_params()==m.EXPECTED_PARAMS[task['variant']],'Final model size mismatch')
    check_dropout(tf,model,task['dropout_rate']);checkpoint_rates(folder/'final.keras',task['dropout_rate'])
    m.require(r['epochs_completed']==task['epochs'],'Incomplete final epoch count')
    expected_steps=int(np.ceil(len(X)/task['batch_size']))*task['epochs']
    m.require(r['optimizer_iterations']==int(model.optimizer.iterations.numpy())==expected_steps,'Final optimizer update count mismatch')
    used=r['history']['learning_rate_used']
    m.require(len(used)==task['epochs'] and np.allclose(used,task['learning_rate_by_epoch'],rtol=1e-6,atol=0),'Final LR schedule mismatch')
    with np.load(folder/'train_probe.npz',allow_pickle=False) as z:
        m.require(np.array_equal(z['indices'],np.arange(32)),'Probe indices changed')
        actual=np.asarray(model(X[:32],training=False)).reshape(-1)
        m.require(np.allclose(actual,z['probability'],rtol=1e-6,atol=1e-7),'Final reload probe mismatch')
    del model
    return r

def fit_task(tf,m,run,task,protocol_sha,X,y):
    m.require(task['batch_size'] in (32,64),'Unsupported batch size')
    folder=run/Path(task['model_relative_path']).parent;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final_report.json').exists() and read(folder/'final_report.json').get('status')=='COMPLETED':
        r=verify_completed(tf,m,folder,task,protocol_sha,X);print('REUSE '+task['key'],flush=True);return r
    print('START '+task['key']+' epochs='+str(task['epochs']),flush=True)
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(task['seed']);m.VARIANT=task['variant'];m.SEED=task['seed']
    model=m.build_model(tf)
    if task['dropout_rate']==.25:set_dropout(tf,model,.25)
    check_dropout(tf,model,task['dropout_rate'])
    options=tf.data.Options();options.experimental_deterministic=True
    options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=1
    weights=np.ones(len(y),dtype=np.float32)
    ds=tf.data.Dataset.from_tensor_slices((X,y.astype(np.float32),weights))
    ds=ds.shuffle(len(y),seed=task['seed'],reshuffle_each_iteration=True).batch(task['batch_size']).with_options(options).prefetch(1)
    r={'status':'RUNNING','task':task,'official_test_read':False,'protocol_sha256':protocol_sha,'epochs_completed':0}
    write(folder/'final_report.json',r)
    class Audit(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):self.lr=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        def on_train_batch_end(self,batch,logs=None):
            m.require(logs is None or all(np.isfinite(float(v)) for v in logs.values()),'Non-finite training metric')
        def on_epoch_end(self,epoch,logs=None):
            m.require(logs and all(np.isfinite(float(v)) for v in logs.values()),'Non-finite epoch metric')
            logs['learning_rate_used']=self.lr
            r['epochs_completed']=epoch+1;r['latest_metrics']={k:float(v) for k,v in logs.items()};write(folder/'final_report.json',r)
    started=time.perf_counter()
    try:
        lr=tf.keras.callbacks.LearningRateScheduler(lambda epoch,previous:task['learning_rate_by_epoch'][epoch])
        h=model.fit(ds,epochs=task['epochs'],callbacks=[lr,Audit(),tf.keras.callbacks.CSVLogger(str(folder/'epoch_log.csv'))],verbose=2)
        r['training_seconds']=time.perf_counter()-started
        before=np.asarray(model(X[:32],training=False)).reshape(-1)
        save_model(model,folder/'final.keras')
        np.savez(folder/'train_probe.npz',indices=np.arange(32,dtype=np.int64),probability=before)
        checkpoint_rates(folder/'final.keras',task['dropout_rate'])
        loaded=tf.keras.models.load_model(folder/'final.keras',safe_mode=True)
        check_dropout(tf,loaded,task['dropout_rate'])
        after=np.asarray(loaded(X[:32],training=False)).reshape(-1)
        m.require(np.allclose(before,after,rtol=1e-6,atol=1e-7),'Final save/reload mismatch')
        expected_steps=int(np.ceil(len(y)/task['batch_size']))*task['epochs']
        m.require(int(model.optimizer.iterations.numpy())==int(loaded.optimizer.iterations.numpy())==expected_steps,'Optimizer update count mismatch')
        r.update(status='COMPLETED',optimizer_iterations=expected_steps,history={k:[float(v) for v in vs] for k,vs in h.history.items()},
            checkpoint_sha256=m.digest(folder/'final.keras'),probe_sha256=m.digest(folder/'train_probe.npz'),
            parameters={'total':model.count_params(),'trainable':int(sum(np.prod(w.shape) for w in model.trainable_weights))},
            reload_max_probability_difference=float(np.max(abs(before-after))))
        m.require(len(r['history']['loss'])==task['epochs'] and np.allclose(r['history']['learning_rate_used'],task['learning_rate_by_epoch'],rtol=1e-6,atol=0),'Fixed schedule not followed')
        write(folder/'final_report.json',r);print('DONE '+task['key'],flush=True)
        del model,loaded,ds;tf.keras.backend.clear_session();gc.collect();return r
    except (Exception,KeyboardInterrupt) as exc:
        r.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=str(exc),traceback=traceback.format_exc())
        write(folder/'final_report.json',r);raise

LABELS={
'A':'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU',
'B':'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'}

def events(m,code,flash,nt):
    m.require(code.shape==(nt,) and np.isin(code,np.arange(13)).all(),'Invalid codes')
    start=m.rising(code)
    m.require(len(start)==180 and np.all(np.diff(start)==42),'Expected 180 flashes with 42-sample onset spacing')
    m.require(flash.shape==code.shape and np.isin(flash,[0,1]).all() and np.array_equal(start,m.rising(flash)),'Flashing mismatch')
    active=(code[1:]>0)&(code[:-1]>0)
    m.require(not np.any(active&(code[1:]!=code[:-1])),'Code changes inside pulse')
    m.require(np.all(start+m.N_RAW<=nt),'Incomplete epochs; no silent removal')
    c=code[start].astype(np.int16)
    for i in range(15):m.require(np.array_equal(np.sort(c[i*12:(i+1)*12]),np.arange(1,13)),'Invalid flash block')
    return start,c

def test_data(m,path):
    mat=loadmat(path,variable_names=['Signal','StimulusCode','Flashing'])
    signal=mat['Signal'];code=mat['StimulusCode'];flash=mat['Flashing']
    m.require(signal.ndim==3 and signal.shape[0]==100 and signal.shape[2]==64,'Expected 100 Test characters /64 channels')
    m.require(code.shape==flash.shape==signal.shape[:2],'Test event dimensions differ')
    X=np.empty((18000,64,52,1),np.float32);meta=np.empty((18000,4),np.int32)
    for char in range(100):
        raw=np.asarray(signal[char],dtype=np.float64);m.require(np.isfinite(raw).all(),'Invalid EEG')
        starts,codes=events(m,code[char],flash[char],len(raw));proc=m.filtered(raw-raw.mean(axis=1,keepdims=True))
        for j,onset in enumerate(starts):
            k=char*180+j;X[k,:,:,0]=m.epoch_at(proc,onset)[0];meta[k]=[char,j//12,codes[j],onset]
        if (char+1)%20==0:print('Test preprocessing',char+1,'/100',flush=True)
    return X,meta

def truth(m,text,meta):
    m.require(len(text)==100 and all(c in m.MATRIX for c in text),'Invalid official label string')
    ids=np.array([m.MATRIX.index(c) for c in text]);idx=ids[meta[:,0]]
    return ((meta[:,2]==idx%6+1)|(meta[:,2]==idx//6+7)).astype(np.int8)

def schedule_from_reports(reports):
    require(len(reports) in (5,15),'Expected 5 or 15 inner-selected histories')
    best=[];rates=[]
    for r in reports:
        h=r['history'];v=np.asarray(h['val_loss'],float);lr=np.asarray(h['learning_rate_used'],float)
        require(len(v)==len(lr)==r['epochs_completed'] and 1<=r['best_epoch']<=len(v)<=200,'Invalid history length')
        require(np.isfinite(v).all() and np.isfinite(lr).all() and (lr>0).all(),'Invalid history values')
        require(int(np.argmin(v))+1==r['best_epoch'] and min(v)==r['best_inner_loss'],'Best epoch was not minimum inner BCE')
        require(np.all(np.diff(lr)<=0),'Learning rate increased unexpectedly')
        best.append(r['best_epoch']);rates.append(lr)
    epochs=int(np.median(best))
    schedule=[float(np.median([lr[min(i,len(lr)-1)] for lr in rates])) for i in range(epochs)]
    return {'epochs':epochs,'learning_rate_by_epoch':schedule,'cv_best_epochs':best,'source_histories':len(reports),
            'source_histories_extended_at_end':sum(len(lr)<epochs for lr in rates)}

# Public workflow. All paths are provided by the caller; dataset geometry is fixed.
TRAIN_HASHES={'A':'462c5fe408ac8b4250075c294b82759adf136f5ea4123b59d01567e1621263f4',
              'B':'dddde80ed7c043850ce3c8e4d7ba737bce3021d5b2f1963a65832b78369b7f84'}
TEST_HASHES={'A':'c432cd6af500a40e01529dd958fc15da3c16adf8115654500aa26e5616b9382b',
             'B':'ac0c48627f08e387a1691215036d43ab1fe7a33228765cb9d1df23e842921922'}


def runtime(threads):
    os.environ['TF_DETERMINISTIC_OPS']='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    return tf, {'python':platform.python_version(),'tensorflow':tf.__version__,
        'keras':getattr(tf.keras,'__version__','unknown'),'numpy':np.__version__,
        'scipy':scipy.__version__,'devices':[d.name for d in tf.config.list_physical_devices()]}


def configuration(profile):
    require(profile in ('primary','ablation'),'Unknown profile')
    return {'profile':profile,'dropout_rate':.25 if profile=='primary' else .5,
        'variants':['none','full'] if profile=='primary' else ['none','channel','temporal','full'],
        'seeds':list(SEEDS),'batch_size':32,'initial_learning_rate':.001,
        'max_epochs':200,'early_stop_patience':20,'class_weights':{'0':1,'1':1},
        'splits':make_splits(),'selection':'earliest minimum inner binary cross-entropy',
        'final_fit':'median epoch and epoch-wise median learning rate from 15 inner histories',
        'decoder':'sum probabilities by stimulus code','primary_repetition':15}


def training_data(data,run,subject):
    path=data/f'Subject_{subject}_Train.mat'
    require(digest(path)==TRAIN_HASHES[subject],'Unexpected benchmark Train file: '+subject)
    return load_raw(path,run/'cache'/subject)


def initialize_run(args,env):
    p={'configuration':configuration(args.profile),'source_sha256':TRAIN_HASHES,
       'code_sha256':digest(Path(__file__)),'environment':env,'threads':args.threads}
    path=args.run_dir/'protocol.json'
    if path.exists():require(read(path)==p,'Run configuration/code/runtime changed; use another run directory')
    else:write(path,p)
    return p


def task_summary(jobs,partition='outer'):
    result={}
    for subject in sorted({j['subject'] for j in jobs}):
        result[subject]={}
        for variant in sorted({j['variant'] for j in jobs}):
            per=[]
            for seed in SEEDS:
                selected=[j for j in jobs if (j['subject'],j['variant'],j['seed'])==(subject,variant,seed)]
                require(len(selected)==5 and {j['fold'] for j in selected}==set(range(1,6)),'Incomplete CV group')
                curves=[j[partition+'_character_curve'] for j in selected]
                curve=[]
                for n in range(15):
                    count=sum(c[n]['correct'] for c in curves);total=sum(c[n]['characters'] for c in curves)
                    curve.append({'sequences':n+1,'correct':count,'characters':total,'accuracy':count/total})
                per.append({'seed':seed,'character_curve':curve,'accuracy_15':curve[-1]['accuracy'],
                            'mean_accuracy_1_to_15':float(np.mean([c['accuracy'] for c in curve]))})
            result[subject][variant]={'per_seed':per}
            for metric in ('accuracy_15','mean_accuracy_1_to_15'):
                values=[x[metric] for x in per]
                result[subject][variant][metric]={'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
    return result


def verify_cv_job(folder,job,protocol_sha):
    require(job['status']=='COMPLETED' and job['protocol_sha256']==protocol_sha,'CV job identity mismatch')
    require(digest(folder/'best.keras')==job['checkpoint_sha256'],'CV checkpoint changed')
    checkpoint_rates(folder/'best.keras',job['dropout_rate'])
    for part in ('inner','outer'):
        f=folder/(part+'_predictions.npz')
        require(digest(f)==job[part+'_predictions_sha256'],'CV prediction changed')
        with np.load(f,allow_pickle=False) as z:
            curve,rows=character_metrics(z['truth'],z['probability'],z['metadata'])
        require(curve==job[part+'_character_curve'] and rows==read(folder/(part+'_characters.json')),'CV score changed')
    return job


def cross_validate(tf,args,protocol):
    m=sys.modules[__name__];config=protocol['configuration'];jobs=[];ph=digest(args.run_dir/'protocol.json')
    for subject in 'AB':
        X,y,meta,_=training_data(args.data_dir,args.run_dir,subject)
        for seed in SEEDS:
            for variant in config['variants']:
                for split in config['splits']:
                    key=f'{subject}_{variant}_{seed}_fold{split["fold"]}'
                    folder=args.run_dir/'cv'/key;folder.mkdir(parents=True,exist_ok=True)
                    task={'subject':subject,'variant':variant,'seed':seed,'fold':split['fold'],'dropout_rate':config['dropout_rate']}
                    print('CV',key,flush=True)
                    if (folder/'report.json').exists():
                        j=read(folder/'report.json')
                        require(all(j[k]==v for k,v in task.items()),'CV task changed')
                        verify_cv_job(folder,j,ph)
                    else:
                        arrays,idx,stats=prepare_fold(X,meta,split)
                        j=train_cv_fold(tf,m,arrays,idx,y,meta,variant,folder,stats,seed,config['dropout_rate'])
                        model=tf.keras.models.load_model(folder/'best.keras',safe_mode=True)
                        prob=predict_batches(model,arrays['outer']);curve,rows=character_metrics(y[idx['outer']],prob,meta[idx['outer']])
                        np.savez_compressed(folder/'outer_predictions.npz',probability=prob,truth=y[idx['outer']],metadata=meta[idx['outer']],indices=idx['outer'])
                        write(folder/'outer_characters.json',rows)
                        j.update(**task,status='COMPLETED',protocol_sha256=ph,outer_character_curve=curve,
                            outer_binary=binary_metrics(y[idx['outer']],prob),
                            inner_predictions_sha256=digest(folder/'inner_predictions.npz'),outer_predictions_sha256=digest(folder/'outer_predictions.npz'))
                        write(folder/'report.json',j)
                        del model,arrays;tf.keras.backend.clear_session();gc.collect()
                    jobs.append(j)
        del X;gc.collect()
    summary={'status':'CV_COMPLETED','protocol_sha256':ph,'jobs':jobs,'summary':task_summary(jobs)}
    write(args.run_dir/'cv_summary.json',summary)
    return summary


def train_final(tf,args,protocol):
    require(args.profile=='primary','The ablation profile is CV-only; use primary for final training')
    cv=read(args.run_dir/'cv_summary.json');ph=digest(args.run_dir/'protocol.json')
    require(cv['status']=='CV_COMPLETED' and cv['protocol_sha256']==ph,'Complete matching CV first')
    expected={(a,v,se,f) for a in 'AB' for v in ('none','full') for se in SEEDS for f in range(1,6)}
    require(len(cv['jobs'])==60 and {(j['subject'],j['variant'],j['seed'],j['fold']) for j in cv['jobs']}==expected,'Incomplete CV source')
    for j in cv['jobs']:
        verify_cv_job(args.run_dir/'cv'/f'{j["subject"]}_{j["variant"]}_{j["seed"]}_fold{j["fold"]}',j,ph)
    tasks=[];normalization={}
    for subject in 'AB':
        raw,_,_,_=training_data(args.data_dir,args.run_dir,subject)
        mean=float(raw.mean(dtype=np.float64));std=float(raw.std(dtype=np.float64));require(std>1e-12,'Invalid scale')
        normalization[subject]={'mean':mean,'std':std,'epsilon':1e-6,'scale':std+1e-6,'fit_on':'all 85 TRAIN characters; never Test'}
        del raw
        for variant in ('none','full'):
            reports=sorted([j for j in cv['jobs'] if j['subject']==subject and j['variant']==variant],key=lambda j:(SEEDS.index(j['seed']),j['fold']))
            recipe=schedule_from_reports(reports)
            for seed in SEEDS:
                key=f'{subject}_{variant}_{seed}'
                tasks.append({'subject':subject,'variant':variant,'seed':seed,'key':key,'dropout_rate':.25,
                    'batch_size':32,'model_relative_path':f'models/{key}/final.keras',**recipe})
    root=args.run_dir/'final';root.mkdir(exist_ok=True)
    frozen={'tasks':tasks,'normalization':normalization,'source_sha256':TRAIN_HASHES,'environment':protocol['environment'],
            'cv_summary_sha256':digest(args.run_dir/'cv_summary.json'),'code_sha256':digest(Path(__file__)),
            'profile':'primary','test_used_for_training':False}
    if (root/'protocol.json').exists():require(read(root/'protocol.json')==frozen,'Frozen training recipe changed')
    else:write(root/'protocol.json',frozen)
    jobs=[];m=sys.modules[__name__]
    for subject in 'AB':
        raw,y,meta,_=training_data(args.data_dir,args.run_dir,subject)
        X=np.array(raw,dtype=np.float32,copy=True);stats=normalization[subject]
        X-=stats['mean'];X/=stats['scale'];del raw;validate_partition(X,y,meta,85)
        for task in tasks:
            if task['subject']==subject:jobs.append(fit_task(tf,m,root,task,digest(root/'protocol.json'),X,y))
        del X;gc.collect()
    manifest={'status':'FINAL_TRAINING_COMPLETED','protocol':frozen,'jobs':jobs}
    write(root/'manifest.json',manifest)


def check_models(root):
    manifest=read(root/'manifest.json');p=manifest['protocol']
    require(manifest['status']=='FINAL_TRAINING_COMPLETED' and p==read(root/'protocol.json'),'Incomplete model manifest')
    require(not p['test_used_for_training'],'Test used in training')
    expected={(a,v,se) for a in 'AB' for v in ('none','full') for se in SEEDS}
    require(len(p['tasks'])==len(manifest['jobs'])==12,'Expected 12 final models')
    require({(t['subject'],t['variant'],t['seed']) for t in p['tasks']}==expected,'Invalid model coverage')
    jobs={j['task']['key']:j for j in manifest['jobs']};require(len(jobs)==12,'Duplicate models')
    for task in p['tasks']:
        key=task['key'];rel=Path(task['model_relative_path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==f'models/{key}/final.keras','Invalid model path')
        d=root/rel.parent;j=jobs[key]
        require(j['task']==task and j['status']=='COMPLETED' and not j['official_test_read'],'Incomplete task')
        require(read(d/'final_report.json')==j and j['protocol_sha256']==digest(root/'protocol.json'),'Model report mismatch')
        require(digest(d/'final.keras')==j['checkpoint_sha256'] and digest(d/'train_probe.npz')==j['probe_sha256'],'Model artifact changed')
        checkpoint_rates(d/'final.keras',task['dropout_rate'])
        require(j['epochs_completed']==task['epochs'],'Epoch count mismatch')
        require(np.allclose(j['history']['learning_rate_used'],task['learning_rate_by_epoch'],rtol=1e-6,atol=0),'Learning rate schedule mismatch')
    return p,jobs


def evaluate(tf,args,env,cross=False):
    root=(args.model_dir or args.run_dir/'final').resolve();p,bykey=check_models(root)
    m=sys.modules[__name__];mode='cross_subject' if cross else 'within_subject'
    out=args.run_dir/'evaluation'/mode;out.mkdir(parents=True,exist_ok=True)
    for target in 'AB':require(digest(args.data_dir/f'Subject_{target}_Test.mat')==TEST_HASHES[target],'Unexpected Test file')
    identity={'mode':mode,'model_manifest_sha256':digest(root/'manifest.json'),'code_sha256':digest(Path(__file__)),
              'environment':env,'test_sha256':TEST_HASHES,'labels_sha256':hashlib.sha256(json.dumps(LABELS,sort_keys=True).encode()).hexdigest()}
    if (out/'identity.json').exists():require(read(out/'identity.json')==identity,'Evaluation identity changed')
    else:write(out/'identity.json',identity)
    jobs=[]
    for source in 'AB':
        target=('B' if source=='A' else 'A') if cross else source
        raw,_,_,_=training_data(args.data_dir,args.run_dir,source);stats=p['normalization'][source]
        require(np.isclose(float(raw.mean(dtype=np.float64)),stats['mean'],rtol=0,atol=1e-10) and np.isclose(float(raw.std(dtype=np.float64)),stats['std'],rtol=1e-12),'Source normalization mismatch')
        probe=np.array(raw[:32],dtype=np.float32,copy=True);probe-=stats['mean'];probe/=stats['scale'];del raw
        X,meta=test_data(m,args.data_dir/f'Subject_{target}_Test.mat');X-=stats['mean'];X/=stats['scale']
        require(np.isfinite(X).all(),'Invalid normalized Test')
        for task in p['tasks']:
            if task['subject']!=source:continue
            key=task['key'];folder=root/Path(task['model_relative_path']).parent;dest=out/key;dest.mkdir(exist_ok=True)
            tf.keras.backend.clear_session();model=tf.keras.models.load_model(folder/'final.keras',safe_mode=True)
            check_dropout(tf,model,task['dropout_rate']);require(model.count_params()==EXPECTED_PARAMS[task['variant']],'Model size mismatch')
            with np.load(folder/'train_probe.npz',allow_pickle=False) as z:
                require(np.array_equal(z['indices'],np.arange(32)),'Probe indices changed');expected=z['probability'].copy()
            actual=np.asarray(model(probe,training=False)).reshape(-1)
            require(np.allclose(actual,expected,rtol=1e-6,atol=1e-7),'Checkpoint inference differs from stored training probe')
            diff=float(np.max(abs(actual-expected)));r=None
            if (dest/'report.json').exists():
                r=read(dest/'report.json');require(r['task']==task and r['source']==source and r['target']==target and r['checkpoint_sha256']==bykey[key]['checkpoint_sha256'],'Saved evaluation identity mismatch')
                require(digest(dest/'predictions.npz')==r['predictions_sha256'],'Saved prediction changed')
                with np.load(dest/'predictions.npz',allow_pickle=False) as z:
                    prob=z['probability'].copy();require(np.array_equal(z['metadata'],meta),'Saved metadata changed')
            else:
                prob=predict_batches(model,X)
                np.savez_compressed(dest/'predictions.npz',probability=prob,metadata=meta)
            # Label strings are used for scoring only after predictions are stored.
            y=truth(m,LABELS[target],meta);curve,rows=character_metrics(y,prob,meta);binary=binary_metrics(y,prob)
            if r is not None:require(r['character_curve']==curve and r['binary']==binary and read(dest/'characters.json')==rows,'Saved score changed')
            else:
                r={'task':task,'source':source,'target':target,'normalization_source':source,'mode':mode,
                   'checkpoint_sha256':bykey[key]['checkpoint_sha256'],'predictions_sha256':digest(dest/'predictions.npz'),
                   'reload_probe_max_difference':diff,'character_curve':curve,'binary':binary}
                write(dest/'characters.json',rows);write(dest/'report.json',r)
            jobs.append(r);print(mode,key,source+'->'+target,'correct/100:',curve[-1]['correct'],flush=True)
            del model;gc.collect()
        del X,probe;gc.collect()
    require(len(jobs)==12,'Incomplete evaluation')
    rows=[]
    for source in 'AB':
        for variant in ('none','full'):
            js=sorted([j for j in jobs if j['source']==source and j['task']['variant']==variant],key=lambda j:SEEDS.index(j['task']['seed']))
            require([j['task']['seed'] for j in js]==list(SEEDS),'Incomplete seed set')
            values=[j['character_curve'][-1]['correct'] for j in js]
            rows.append({'source':source,'target':js[0]['target'],'variant':variant,'seeds':list(SEEDS),'correct_out_of_100':values,
                'mean_accuracy_percent':float(np.mean(values)),'sample_sd_percentage_points':float(np.std(values,ddof=1)),
                'mean_accuracy_1_to_15_percent':100*float(np.mean([c['accuracy'] for j in js for c in j['character_curve']]))})
    write(out/'summary.json',{'status':'EVALUATION_COMPLETED','identity':identity,'jobs':jobs,'summary':rows})
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def self_test():
    splits=make_splits();require(len(splits)==5,'Wrong fold count')
    require(sorted(c for s in splits for c in s['outer_characters'])==list(range(85)),'OOF coverage')
    meta=np.array([[g,s,c,42*(s*12+c-1)] for g in range(100) for s in range(15) for c in range(1,13)],np.int32)
    for sub in 'AB':
        y=truth(sys.modules[__name__],LABELS[sub],meta);require(y.sum()==3000,'Target mapping')
        curves,_=character_metrics(y,np.where(y,.9,.1),meta);require(all(c['correct']==100 for c in curves),'Decoder mapping')
    rng=np.random.default_rng(42);raw=rng.normal(size=(8000,64));processed=filtered(raw-raw.mean(axis=1,keepdims=True))
    for onset in (0,42,7518):require(epoch_at(processed,onset)[0].shape==(64,52),'Epoch shape')
    reports=[{'best_epoch':i+1,'best_inner_loss':0.,'epochs_completed':20,'history':{'val_loss':[0. if k==i else 1. for k in range(20)],'learning_rate_used':[.001]*20}} for i in range(15)]
    require(schedule_from_reports(reports)['epochs']==8,'Median recipe')
    print('DATA_SELF_TEST_PASSED (no TensorFlow training/inference tested)')


def verify_release(home):
    manifest=read(home/'SHA256.json')
    for name,h in manifest.items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Invalid manifest path')
        require(digest(home/rel)==h,'Release file changed: '+name)
    for root in sorted((home/'references').iterdir()):
        if (root/'manifest.json').exists():check_models(root)
    print('RELEASE_FILES_VERIFIED')


def smoke_test(tf):
    """Exercise each graph, one optimization step, and Keras serialization."""
    import tempfile
    global VARIANT
    rng=np.random.default_rng(2026)
    X=rng.normal(size=(4,64,52,1)).astype(np.float32)
    y=np.array([0,1,0,1],np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        for variant in EXPECTED_PARAMS:
            for rate in (.25,.5):
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
                VARIANT=variant;model=build_model(tf)
                if rate==.25:set_dropout(tf,model,rate)
                metrics=model.train_on_batch(X,y)
                require(np.isfinite(metrics).all(),'Non-finite smoke-test training')
                before=predict_batches(model,X)
                path=Path(temporary)/'model.keras';save_model(model,path)
                loaded=tf.keras.models.load_model(path,safe_mode=True)
                check_dropout(tf,loaded,rate);checkpoint_rates(path,rate)
                require(np.allclose(before,predict_batches(loaded,X),rtol=1e-6,atol=1e-7),'Serialization mismatch')
                require(loaded.count_params()==EXPECTED_PARAMS[variant],'Parameter mismatch')
                print('PASS',variant,'dropout',rate)
    print('TENSORFLOW_SMOKE_TEST_PASSED; this is not a full scientific reproduction')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['self-test','smoke-test','verify','preprocess','cv','train','test','transfer','all'])
    ap.add_argument('--data-dir',type=Path,help='Directory containing the original Subject_A/B_Train/Test.mat files')
    ap.add_argument('--run-dir',type=Path,help='Writable directory for this protocol and its outputs')
    ap.add_argument('--model-dir',type=Path,help='Frozen model bundle for test/transfer; defaults to RUN/final')
    ap.add_argument('--profile',choices=['primary','ablation'],default='primary')
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.command=='self-test':self_test();return 0
    if args.command=='verify':verify_release(Path(__file__).resolve().parent);return 0
    if args.command=='smoke-test':
        tf,_=runtime(args.threads);smoke_test(tf);return 0
    if args.data_dir is None or args.run_dir is None:ap.error('--data-dir and --run-dir are required')
    require(args.threads>0,'Threads must be positive')
    args.data_dir=args.data_dir.resolve();args.run_dir=args.run_dir.resolve()
    require(args.run_dir!=args.data_dir and args.run_dir not in args.data_dir.parents and args.data_dir not in args.run_dir.parents,'Data and output directories must not overlap')
    if args.model_dir:
        args.model_dir=args.model_dir.resolve()
        require(args.run_dir!=args.model_dir and args.run_dir not in args.model_dir.parents and args.model_dir not in args.run_dir.parents,'External model bundle and output must not overlap')
    if args.profile=='ablation' and args.command not in ('preprocess','cv'):ap.error('Ablation profile supports preprocess/cv only')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    lock=args.run_dir/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        if args.command=='preprocess':
            for sub in 'AB':training_data(args.data_dir,args.run_dir,sub)
            return 0
        tf,env=runtime(args.threads)
        if args.command in ('cv','train','all'):
            protocol=initialize_run(args,env)
            if args.command in ('cv','all'):cross_validate(tf,args,protocol)
            if args.command in ('train','all'):train_final(tf,args,protocol)
        if args.command in ('test','all'):evaluate(tf,args,env)
        if args.command=='transfer':evaluate(tf,args,env,cross=True)
        print('COMPLETED:',args.command)
    except BaseException as exc:
        write(args.run_dir/'last_error.json',{'command':args.command,'error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:lock.unlink(missing_ok=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
