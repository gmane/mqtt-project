# Load forecast project TSERI
# script intended for neural network library (https://code.google.com/p/neurolab/)


##------------------------------------------------------- Library

def loadforecast(window,model,path,datafile):
    
 import numpy as np

 import math

 from numpy import genfromtxt

 import os

 import time

##------------------------------------------------------- Parameters

 pred_off=window

 ModelNumber=model#[1,2,3,5]

 ModelNumber=np.asarray(ModelNumber)

 stWeekday=1
 stHour=23

 nLower=-0.5
 nUpper=0.5

 loadFilterlength=2

 # CHANGE NETWORK STRUCTURE 
 neuron=5

 ##------------------------------------------------------- Load data
 starttime=time.time()

 dataR = genfromtxt(datafile, delimiter=',')

 if np.ndim(dataR)==1:
    dataR=dataR.reshape(dataR.shape[0],1) # Python cann't find both dimenstions of 1D array

 if dataR.shape[0]>dataR.shape[1]:
    dataR=dataR.T

 dataR=dataR[:,1:dataR.shape[1]] # removed header
 newdataR=np.zeros((2,dataR.shape[1]/4))
 for i in range(0,dataR.shape[1]-1,4):
    newdataR[0,i/4]=dataR[0,i]
    newdataR[1,i/4]=dataR[1,i]
    
##print dataR.shape
##dataR=dataR[:,0:1580]
##print dataR.shape

# load weather data
 if pred_off==0.15:
    wForecast=genfromtxt('WeatherForecast015.csv', delimiter=',')
 if pred_off==1:
    dataR=newdataR
    wForecast=genfromtxt('WeatherForecast1.csv', delimiter=',')
 if pred_off==24:
    dataR=newdataR
    wForecast=genfromtxt('WeatherForecast24.csv', delimiter=',')

 if np.ndim(wForecast)==1:
    wForecast=wForecast.reshape(wForecast.shape[0],1) # Python cann't find both dimenstions of 1D array
    
 wForecast=wForecast[1:pred_off+1] # remove header & excess samples are omitted  

  
##------------------------------------------------------- Filter load data

 def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


 loadF=movingaverage(dataR[0,:],loadFilterlength)

### to plot and check manually
##plt.plot(dataR[0,:],'b.-', label="Raw")
##plt.plot(loadF,'r.-',label="Moving average")
##plt.xlabel('Time (hours)')
##plt.ylabel('Energy Consumption (Wh)')
##plt.title('X hour ahead energy forecast')
##plt.legend()
##plt.show()

##------------------------------------------------------- Input for forecast window

 data=np.vstack((loadF,dataR[1:dataR.shape[0],:])) # Model data start
 predMatrix=np.zeros((dataR.shape[0],pred_off))

# Add forecasts to the inputs in forecast window
 predMatrix[predMatrix.shape[0]-1,:]=wForecast.T
 data=np.concatenate((data,predMatrix),1)
##print(data.shape)

##data[1,data.shape[1]-pred_off:data.shape[1]]=wForecast.T #####################

##------------------------------------------------------- Date time

 if pred_off==1:
   gentime=np.zeros((2,data.shape[1]))
   gentime[0,0]=stWeekday
   gentime[1,0]=stHour
   for i in range(0,data.shape[1]-1):
    
        gentime[0,i+1]=gentime[0,i]
        gentime[1,i+1]=gentime[1,i]+1
        
        if gentime[1,i]==24:
            gentime[0,i+1]=gentime[0,i]+1
            gentime[1,i+1]=1
            
        if gentime[0,i]==7 and gentime[1,i]==24:
            gentime[0,i+1]=1
   data=np.vstack((data,gentime))

 if pred_off==24:
   gentime=np.zeros((2,data.shape[1]))
   gentime[0,0]=stWeekday
   gentime[1,0]=stHour
   for i in range(0,data.shape[1]-1):
    
        gentime[0,i+1]=gentime[0,i]
        gentime[1,i+1]=gentime[1,i]+1
        
        if gentime[1,i]==24:
            gentime[0,i+1]=gentime[0,i]+1
            gentime[1,i+1]=1
            
        if gentime[0,i]==7 and gentime[1,i]==24:
            gentime[0,i+1]=1
   data=np.vstack((data,gentime))

 if pred_off==0.15:
   gentime=np.zeros((2,data.shape[1]))
   gentime[0,0]=stWeekday
   gentime[1,0]=stHour
   for i in range(0,data.shape[1]-1):
    
        gentime[0,i+1]=gentime[0,i]
        gentime[1,i+1]=gentime[1,i]+1
        
        if gentime[1,i]==24*4:
            gentime[0,i+1]=gentime[0,i]+1
            gentime[1,i+1]=1
            
        if gentime[0,i]==7*4 and gentime[1,i]==24*4:
            gentime[0,i+1]=1
   data=np.vstack((data,gentime))

##------------------------------------------------------- Hour ahead input
 if pred_off==0.15:
    numInput=4
    genInp=np.zeros((numInput,data.shape[1]))
    # Type-1
    genInp[0,1:data.shape[1]]=data[0,0:data.shape[1]-1]
    # Type-2
    genInp[1,2:data.shape[1]]=data[0,0:data.shape[1]-2]
    # Type-3
    genInp[2,3:data.shape[1]]=data[0,0:data.shape[1]-3]
    # Type-4
    genInp[3,4:data.shape[1]]=data[0,0:data.shape[1]-4]
    data=np.vstack((data,genInp))
    
 if pred_off==1:
    
    numInput=5
    genInp=np.zeros((numInput,data.shape[1]))
    # Type-1
    genInp[0,1:data.shape[1]]=data[0,0:data.shape[1]-1]
    # Type-2
    genInp[1,2:data.shape[1]]=data[0,0:data.shape[1]-2]
    # Type-3
    genInp[2,3:data.shape[1]]=data[0,0:data.shape[1]-3]
    # Type-4
    genInp[3,4:data.shape[1]]=data[0,0:data.shape[1]-4]
    # Type-5
    genInp[4,5:data.shape[1]]=data[0,0:data.shape[1]-5]


    data=np.vstack((data,genInp))


    # previous hours average
    m=np.zeros((4,data.shape[1]))

    for i in np.arange(2,data.shape[1],1):
        m[0,i]=np.mean((data[0,i-1],data[0,i-2]))
        if i>=3:
                m[1,i]=np.mean((data[0,i-1],data[0,i-2],data[0,i-3]))
        if i>=4:
                m[2,i]=np.mean((data[0,i-1],data[0,i-2],data[0,i-3],data[0,i-4]))
        if i>=5:
                m[3,i]=np.mean((data[0,i-1],data[0,i-2],data[0,i-3],data[0,i-4],data[0,i-5]))

    data=np.vstack((data,m))


 if pred_off==24:
    
    #Previous day same hour 
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,23,-1):
            for j in np.arange(i,i-25,-1):
                    if data[data.shape[0]-2,i]==data[data.shape[0]-2,j] and data[data.shape[0]-3,j]==data[data.shape[0]-3,i]-1:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-3,i]==1:
                            if data[data.shape[0]-2,i]==data[data.shape[0]-2,j] and data[data.shape[0]-3,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]
    
    
    #Previous second day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,47,-1):
            for j in np.arange(i,i-49,-1):
                    if data[data.shape[0]-3,i]==data[data.shape[0]-3,j] and data[data.shape[0]-4,j]==data[data.shape[0]-4,i]-2:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-4,i]==1:
                            if data[data.shape[0]-3,i]==data[data.shape[0]-3,j] and data[data.shape[0]-4,j]==6:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-4,i]==2:
                            if data[data.shape[0]-3,i]==data[data.shape[0]-3,j] and data[data.shape[0]-4,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]


    #Previous third day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,71,-1):
            for j in np.arange(i,i-73,-1):
                    if data[data.shape[0]-4,i]==data[data.shape[0]-4,j] and data[data.shape[0]-5,j]==data[data.shape[0]-5,i]-3:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-5,i]==1:
                            if data[data.shape[0]-4,i]==data[data.shape[0]-4,j] and data[data.shape[0]-5,j]==5:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-5,i]==2:
                            if data[data.shape[0]-4,i]==data[data.shape[0]-4,j] and data[data.shape[0]-5,j]==6:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-5,i]==3:
                            if data[data.shape[0]-4,i]==data[data.shape[0]-4,j] and data[data.shape[0]-5,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]


    #Previous fourth day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,95,-1):
            for j in np.arange(i,i-97,-1):
                    if data[data.shape[0]-5,i]==data[data.shape[0]-5,j] and data[data.shape[0]-6,j]==data[data.shape[0]-6,i]-4:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-6,i]==1:
                            if data[data.shape[0]-5,i]==data[data.shape[0]-5,j] and data[data.shape[0]-6,j]==4:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-6,i]==2:
                            if data[data.shape[0]-5,i]==data[data.shape[0]-5,j] and data[data.shape[0]-6,j]==5:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-6,i]==3:
                            if data[data.shape[0]-5,i]==data[data.shape[0]-5,j] and data[data.shape[0]-6,j]==6:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-6,i]==4:
                            if data[data.shape[0]-5,i]==data[data.shape[0]-5,j] and data[data.shape[0]-6,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]


    #Previous fifth day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,119,-1):
            for j in np.arange(i,i-121,-1):
                    if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==data[data.shape[0]-7,i]-5:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-7,i]==1:
                            if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==3:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-7,i]==2:
                            if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==4:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-7,i]==3:
                            if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==5:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-7,i]==4:
                            if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==6:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-7,i]==5:
                            if data[data.shape[0]-6,i]==data[data.shape[0]-6,j] and data[data.shape[0]-7,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]

    #Previous sixth day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,143,-1):
            for j in np.arange(i,i-145,-1):
                    if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==data[data.shape[0]-8,i]-6:              
                            data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==1:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==2:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==2:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==3:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==3:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==4:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==4:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==5:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==5:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==6:
                                    data[data.shape[0]-1,i]=data[0,j]
                    if data[data.shape[0]-8,i]==6:
                            if data[data.shape[0]-7,i]==data[data.shape[0]-7,j] and data[data.shape[0]-8,j]==7:
                                    data[data.shape[0]-1,i]=data[0,j]


    #Previous same day same hour
    m=np.zeros((1,data.shape[1]))
    data=np.vstack((data,m))
    
    for i in np.arange(data.shape[1]-1,167,-1):
            for j in np.arange(i,i-169,-1):
                    if data[data.shape[0]-8,i]==data[data.shape[0]-8,j] and data[data.shape[0]-9,j]==data[data.shape[0]-9,i]:              
                            data[data.shape[0]-1,i]=data[0,j]


    # privious days mean
    nMeans=6
    m=np.zeros((nMeans,data.shape[1]))

    for i in np.arange(data.shape[1]):
            m[0,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i]))
            m[1,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i],data[data.shape[0]-5,i]))
            m[2,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i],data[data.shape[0]-5,i],data[data.shape[0]-4,i]))
            m[3,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i],data[data.shape[0]-5,i],data[data.shape[0]-4,i],data[data.shape[0]-3,i]))
            m[4,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i],data[data.shape[0]-5,i],data[data.shape[0]-4,i],data[data.shape[0]-3,i],data[data.shape[0]-2,i]))
            m[5,i]=np.mean((data[data.shape[0]-7,i],data[data.shape[0]-6,i],data[data.shape[0]-5,i],data[data.shape[0]-4,i],data[data.shape[0]-3,i],data[data.shape[0]-2,i],data[data.shape[0]-1,i]))

    data=np.vstack((data,m))
    

##np.savetxt("data.csv", data, delimiter=",")
 correlation=np.corrcoef(data[0,:],data[1:data.shape[0],:])
##print("\nCorrelation: ",correlation[0,:])

##------------------------------------------------------- Normalize data

 dataNormalized=np.zeros((data.shape[0],data.shape[1]))

 for i in range(0, data.shape[0]):

    for j in range(0,data.shape[1]):       
        dataNormalized[i,j]=((nUpper-nLower)*(data[i,j]-data[i].min()))/(data[i].max()-data[i].min())+nLower


 inputs=dataNormalized[1:data.shape[0],:]

 inputs=inputs.T # inputs needs to be row oriented

 targets=dataNormalized[0]
 targets=targets.reshape(data.shape[1],1)

 preoff=0
 if pred_off==0.15:
    pred_off=1
    preoff=0.15

 range1=np.arange(7*pred_off,inputs.shape[0]-pred_off)   # removeing first zero elements
 range2=np.arange(inputs.shape[0]-pred_off,inputs.shape[0])

 storeforecast=np.zeros((6,pred_off))

##------------------------------------------------------- Network structure

 for m in np.arange(0,ModelNumber.shape[0]):

    if ModelNumber[m]==1:
        t_01=time.time()
        
        import neurolab as nl

        forecast=[]
        forecast=np.array(forecast)

        while (forecast.shape[0]<pred_off):
            if pred_off==1 and (dataR.shape[0]-1)==0 and preoff==0.15:       
                net = nl.net.newff([[nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper]], [neuron, 1])
            if pred_off==1 and (dataR.shape[0]-1)==1 and preoff==0.15:       
                net = nl.net.newff([[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper]], [neuron, 1])

            if pred_off==1 and (dataR.shape[0]-1)==0 and preoff==0:       
                net = nl.net.newff([[nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper]], [neuron, 1])
            if pred_off==1 and (dataR.shape[0]-1)==1 and preoff==0:       
                net = nl.net.newff([[nLower,nUpper],[nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper]], [neuron, 1])

            if pred_off==24 and (dataR.shape[0]-1)==0 and preoff==0:
                net = nl.net.newff([[nLower,nUpper],[nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper]],[neuron, 1])
            if pred_off==24 and (dataR.shape[0]-1)==1 and preoff==0:
                net = nl.net.newff([[nLower,nUpper],[nLower,nUpper],[nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper]],[neuron, 1])

##            net = nl.net.newff([[nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper], [nLower,nUpper],[nLower,nUpper], [nLower,nUpper], [nLower,nUpper]],[neuron, 1])
            err = net.train(inputs[range1,:], targets[range1,:], epochs=100, show=15, goal=0.02)

            out=net.sim(inputs[range2,:])
            # Forecast can not be negative
            forecast=out
            
        forecast=forecast.T
        storeforecast[0,:]=forecast
        t_02=time.time()
        
    if ModelNumber[m]==2:
        t_03=time.time()
        
        forecast=np.zeros((1,pred_off))
        targetslist=targets.tolist()
        targetslist=[l[0] for l in targetslist]
        inputslist = [0] * inputs.shape[0]

        if pred_off==1 and preoff==0.15:
          for i in range(0, inputs.shape[0]):
         
            inputslist[i]={1: inputs[i,0], 2: inputs[i,1], 3: inputs[i,2], 4: inputs[i,3], 5: inputs[i,4], 6: inputs[i,5], 7: inputs[i,6]}

        if pred_off==1 and preoff==0:
          for i in range(0, inputs.shape[0]):
         
            inputslist[i]={1: inputs[i,0], 2: inputs[i,1], 3: inputs[i,2], 4: inputs[i,3], 5: inputs[i,4], 6: inputs[i,5], 7: inputs[i,6], 8: inputs[i,7], 9: inputs[i,8], 10: inputs[i,9], 11: inputs[i,10], 12: inputs[i,11]}
        if pred_off==24 and preoff==0:
          for i in range(0, inputs.shape[0]):
         
            inputslist[i]={1: inputs[i,0], 2: inputs[i,1], 3: inputs[i,2], 4: inputs[i,3], 5: inputs[i,4], 6: inputs[i,5], 7: inputs[i,6], 8: inputs[i,7], 9: inputs[i,8], 10: inputs[i,9], 11: inputs[i,10], 12: inputs[i,11] ,13: inputs[i,12], 14: inputs[i,13], 15: inputs[i,14], 16: inputs[i,15]}


       # os.chdir ('C:\\Users\\jlk469\\Downloads\\python')
       # os.chdir ('C:\\Users\\jlk469\\Documents\\libsvm-3.20\\python')
        
       # os.chdir("//home//pi//Downloads//libsvm-3.20//python")

        os.chdir("//home//pi//Desktop//rPi_Li//libsvm-3.20//python")
        
        from svmutil import svmutil

        #mode2
        train1=range1[0]
        train2=range1[range1.shape[0]-1]
        predict1=range2[0]
        predict2=range2[range2.shape[0]-1]
                
        y=targetslist[train1:train2+1]
        x=inputslist[train1:train2+1]
        model=svm_train(y,x,'-s 3 -t 2 -g 0.1 -c 0.1 -p 0.001')
        new=inputslist[predict1:predict2+1]
        rand=targetslist[predict1-24-1:predict2-24]
        prediction, p_acc, p_val =svm_predict(rand,new,model)
        for i in range(0,pred_off-1):
             forecast[0,i]=prediction[i]

        storeforecast[1,:]=forecast
        t_04=time.time()
        
    if ModelNumber[m]==3:
        t_05=time.time()
        
        targets=dataNormalized[0]

        from sklearn.svm import SVR
        
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

        y_rbf = svr_rbf.fit(inputs[range1,:], targets[range1]).predict(inputs[range2,:])

        forecast=y_rbf.reshape(y_rbf.shape[0],np.ndim(y_rbf))
        forecast=forecast.T
        storeforecast[2,:]=forecast

        t_06=time.time()
        
    if ModelNumber[m]==4:

        t_07=time.time()

        from sklearn.gaussian_process import GaussianProcess

        gp = GaussianProcess(corr='squared_exponential', theta0=1e-1, thetaL=1e-3, thetaU=1e-1, nugget=1e-2, random_start=100)

        gp.fit(inputs[range1,:], targets[range1])

        y_pred, MSE = gp.predict(inputs[range2,:], eval_MSE=True)
        
        forecast=y_pred.T
        storeforecast[3,:]=forecast

        t_08=time.time()
        
    if ModelNumber[m]==5:
        t_09=time.time()

        targets=targets.reshape(1,targets.shape[0]) # i changed (SM); was working individually but not with others
        
        inputs=inputs.T
        forecast=np.zeros((1,pred_off))
        index=range1[range1.shape[0]-1]
             
        if pred_off==1 and preoff==0.15:
            for i in range(0, 6):
                train1=range1[range1.shape[0]-1]-3-i*4
                train2=range1[range1.shape[0]-1]-i*4
                predict1=range2[range2.shape[0]-1]-3-i*4
                predict2=range2[range2.shape[0]-1]-i*4
 
                y=targets[:,train1:train2+1]
                x=inputs[:,train1:train2+1]
            
                X = np.vstack([np.ones(4), x]).T
                y=y.T
                a1,a2,a3,a4,a5,a6,a7,a8=np.linalg.lstsq(X, y)[0]
                predict=range2[range2.shape[0]-1]
                new=inputs[:,predict]
                forecast[0,0]=a1+new[0]*a2+new[1]*a3+new[2]*a4+new[3]*a5+new[4]*a6+new[5]*a7+new[6]*a8

   
        if pred_off==24 and preoff==0:
            for i in range(0, 6):
                train1=range1[range1.shape[0]-1]-3-i*4
                train2=range1[range1.shape[0]-1]-i*4
                predict1=range2[range2.shape[0]-1]-3-i*4
                predict2=range2[range2.shape[0]-1]-i*4

                y=targets[:,train1:train2+1]
                x=inputs[:,train1:train2+1]
                
                X = np.vstack([np.ones(4), x]).T
                y=y.T
                a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17=np.linalg.lstsq(X, y)[0]
                pre=inputs[:,predict1:predict2+1]
                for j in range(0, 4):
                     new=pre[:,j]
                     prediction=a1+new[0]*a2+new[1]*a3+new[2]*a4+new[3]*a5+new[4]*a6+new[5]*a7+new[6]*a8+new[7]*a9+new[8]*a10+new[9]*a11+new[10]*a12+new[11]*a13+new[12]*a14+new[13]*a15+new[14]*a16+new[15]*a17
                     forecast[0,predict1-index+j-1]=prediction

                storeforecast[4,:]=forecast
                
        if pred_off==1 and preoff==0:
            train1=range1[range1.shape[0]-1]-3
            train2=range1[range1.shape[0]-1]
            x=inputs[:,train1:train2+1]
            y=targets[:,train1:train2+1]
            X = np.vstack([np.ones(4), x]).T
            y=y.T
            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13=np.linalg.lstsq(X, y)[0]
            predict=range2[range2.shape[0]-1]
            new=inputs[:,predict]
            forecast[0,0]=a1+new[0]*a2+new[1]*a3+new[2]*a4+new[3]*a5+new[4]*a6+new[5]*a7+new[6]*a8+new[7]*a9+new[8]*a10+new[9]*a11+new[10]*a12+new[11]*a13

            storeforecast[4,:]=forecast 

        t_10=time.time()

        
##------------------------------------------------------- denormalize data
 dataOriginal=np.zeros((storeforecast.shape[0],storeforecast.shape[1]))

 for i in range(0,storeforecast.shape[0]):
    for j in range(0,storeforecast.shape[1]):
        dataOriginal[i,j]=(((storeforecast[i,j]-nLower)*(data[data.shape[0]-1].max()-data[data.shape[0]-1].min()))/(nUpper-nLower))+data[data.shape[0]-1].min()
                                    
 forecast=dataOriginal

 forecast=forecast.astype(int)


##------------------------------------------------------- Plot

# import matplotlib.pyplot as plt

##plotForecast=plt.figure(1)    
##plt.plot(storeforecast[0,:],'r.-', label="Forecast")
##plt.hold('on')
##plt.plot(storeforecast[2,:],'r.-', label="Forecast")

##plt.plot(forecast[0,:],'r.-', label="Forecast")

# plt.xlabel('Time (hours)',fontsize=20)
# plt.xticks(size = 15)
# plt.ylabel('Energy Consumption (KWh)',fontsize=20)
# plt.yticks(size = 15)

# if pred_off==24:
 #       plt.title('24-hours ahead energy forecast',fontsize=20)
# elif pred_off==1:
#        plt.title('Hour ahead energy forecast',fontsize=20)

##plt.legend()
##plotForecast.show()

##plotForecast1=plt.figure(2)
##plt.plot(np.arange(1,25),forecast[0],'bs-',label="Forecast")
##plotForecast1.show()

##plt.plot(a,b,'r--',np.arange(1,25),forecast[0],'bs')

##plotForecast2=plt.figure(3)
##x=np.arange(-23,1)
##y=dataR[0,dataR.shape[1]-pred_off:dataR.shape[1]]
##plt.plot(x,y,'bo-',label="Today")
##plt.plot(np.arange(1,25),forecast[0],'rs-',label="Forecast")
##plotForecast2.show()


# for m in np.arange(0,ModelNumber.shape[0]):
    
#    if ModelNumber[m]==1:
#        plt.plot(dataOriginal[0,:],'b*-', linewidth=2.0, markersize=15,label="Model-1")
#    if ModelNumber[m]==2:
#        plt.plot(dataOriginal[1,:],'gp-', linewidth=2.0, markersize=12,label="Model-2")
#    if ModelNumber[m]==3:
#        plt.plot(dataOriginal[2,:],'r^-', linewidth=2.0, markersize=12,label="Model-3")
#    if ModelNumber[m]==4:
#        plt.plot(dataOriginal[3,:],'cd-', linewidth=2.0, markersize=12,label="Model-4")
#    if ModelNumber[m]==5:
#        plt.plot(dataOriginal[4,:],'ko-', linewidth=2.0, markersize=12,label="Model-5")
##    if ModelNumber[m]==6:
##        plt.plot(dataOriginal[5,:],'ms-', linewidth=2.0, markersize=12,label="Model-6")

 endtime=time.time()
 elapsedtime=endtime-starttime

# plt.legend()
# plt.show()

 a=path#"//home//pi//Desktop//rPi_Li//libsvm-3.20//python"
 os.chdir(a)
 np.savetxt("Forecast.csv", dataOriginal, delimiter=",")

 return dataOriginal

