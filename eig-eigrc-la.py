import numpy as np
from numpy import linalg as la
import cv2 # pentru citirea pozelor
import matplotlib
from matplotlib import pyplot as plt # pentru grafice

nrPersoane=40
nrPixeli=10304
nrPozeAntrenare=8
nrTotalPozeAntrenare=nrPozeAntrenare*nrPersoane
A=np.zeros([10304,320])
k=40

def nn(A,pozaCautataVect,norm):
    z=np.zeros(len(A[0]))
    for i in range (0,len(A[0])):
        
        if norm=="2":
                z[i]=la.norm(A[:,i]-pozaCautataVect)
            
        elif norm=="1":
                z[i]=la.norm(A[:,i]-pozaCautataVect,1)
            
        elif norm=="infinit":
                z[i]=la.norm(A[:,i]-pozaCautataVect,np.inf)
            
        elif norm=="cos":
                z[i]=1-np.dot(A[:,i],pozaCautataVect)/(la.norm(A[:,i])*la.norm(pozaCautataVect))
        
            
            
    pozitia=np.argmin(z)
            
    return pozitia

def configurareA():
	#AT&T Database of Faces
	caleBD=r'F:\zSCOALA\acs folder\proiect\att_faces' #calea catre folder cu poze de recunoastere
	for i in range(1,nrPersoane+1):
		caleFolderPers=caleBD+'\s'+str(i)+'\\'
		for j in range(1,nrPozeAntrenare+1):
			calePozaAntrenare=caleFolderPers+str(j)+'.pgm'
			# citim poza ca matrice 112 x 92:
			pozaAntrenare=np.array(cv2.imread(calePozaAntrenare,0))
			# vectorizam poza:
			pozaVect=pozaAntrenare.reshape(10304,) 
			A[:,nrPozeAntrenare*(i-1)+(j-1)] = pozaVect

def cautareNN():
    configurareA()  
	# testare poza 9 a persoanei 40:
    calePozaCautata=r'F:\zSCOALA\acs folder\proiect\att_faces\s8\10.pgm'
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVect=pozaCautata.reshape(-1,)
    
    plt.imshow(pozaCautataVect.reshape(112,92), cmap='gray',vmin=0, vmax=255) 
    plt.show()
    
    
    poz=nn(A,pozaCautataVect,"1") # apel algoritm NN cu norma 1
    
    print()
    print("Nr poza",poz)
    
    plt.imshow(A[:,poz].reshape(112,92), cmap='gray',vmin=0, vmax=255)
    plt.show() 
  
  
#--------------------------------------------------------------------------------    


def preprocesare_EF():
    global media_EF
    global proiectare
    global HQPB

    B_ma=A
    
    #poza medie
    media_EF=np.mean(B_ma,axis=1)
    
    #centram
    B_ma=(B_ma.T-media_EF).T
    
    #matrice de covarianta a pixelilor
    C=B_ma.T@B_ma
    #C=np.dot(B_ma.T,B_ma) #functioneza
    
    #vectorii proprii ai matricei C
    D,V_EF=np.linalg.eig(C)
    V_EF=B_ma@V_EF
    indici_EF=D.argsort()
    
  
    HQPB=V_EF[:,indici_EF[-1:-k-1:-1]]
    
    #proiectare
    proiectare=np.dot(B_ma.T,HQPB)


    
    
def EF(PozaCaut):


    #centram
    PozaCaut=PozaCaut-media_EF
    #proiectare poza pe HQPB
    PR_PozaCaut=np.dot(PozaCaut,HQPB)
    
    #cautare cu NN
    pozitia=nn(proiectare.T,PR_PozaCaut,'1')
    return pozitia


def cautareEIG():
    configurareA()
    preprocesare_EF()
    
    calePozaCautata=r'F:\zSCOALA\acs folder\proiect\att_faces\s8\10.pgm'
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVect=pozaCautata.reshape(-1,)

    plt.imshow(pozaCautataVect.reshape(112,92), cmap='gray',vmin=0, vmax=255) 
    plt.show()
    
    poz=EF(pozaCautataVect)    
    print()
    print("Nr poza",poz)
    
    plt.imshow(A[:,poz].reshape(112,92), cmap='gray',vmin=0, vmax=255)
    plt.show() 
    
  
  


#-----------------------------------------------------------------------------------


def preprocesare_EFRC():
    global media_EFRC
    global proiectare_RE
    global HQPBRC
    
    B_ma=A
    RC=np.zeros([10304,40])
    for i in range(0,40):
        j=i*8+7
        RC[:,i]=np.mean(B_ma[:,0:j+1:1],axis=1)
    
    #poza medie
    media_EFRC=np.mean(RC,axis=1)
    
    #centram
    RC=(RC.T-media_EFRC).T
    
    #matrice de covarianta a pixelilor
    C=RC.T@RC
    
    
    #vectorii proprii ai matricei C
    D,V_EFRC=np.linalg.eig(C)
    V_EFRC=RC@V_EFRC
    indici_EFRC=D.argsort()
    
   
    HQPBRC=V_EFRC[:,indici_EFRC[-1:-k-1:-1]]
 

    #proiectare
    proiectare_RE=np.dot(RC.T,HQPBRC)



def EFRC(PozaCaut):


    #centram
    PozaCaut=PozaCaut-media_EFRC
    #proiectare
    PR_PozaCaut=np.dot(PozaCaut,HQPBRC)
    
    #cautare cu NN
    pozitia=nn(proiectare_RE.T,PR_PozaCaut,'1')
    return pozitia

def cautareEIGRC():
    configurareA()
    preprocesare_EFRC()
    
    calePozaCautata=r'F:\zSCOALA\acs folder\proiect\att_faces\s8\10.pgm'
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVect=pozaCautata.reshape(-1,)

    plt.imshow(pozaCautataVect.reshape(112,92), cmap='gray',vmin=0, vmax=255) 
    plt.show()
    
    poz=EFRC(pozaCautataVect)    
    print()
   
    
    plt.imshow(A[:,poz*8-1].reshape(112,92), cmap='gray',vmin=0, vmax=255)
    plt.show() 

#-----------------------------------------------------------------------------------

def LA(PozaCaut):
    global media_LA
    
    B=A
    q = np.zeros([10304,k+2])
    q[:,0]=np.zeros((10304))
    q[:,1]=np.ones((10304))
    q[:,1]=q[:,1]/la.norm(q[:,1])


    b=0
    for i in range (1,k):
        w=np.subtract(np.dot(B,(np.dot(B.T,q[:,i]))),b*q[:,i-1])
        a=np.dot(w,q[:,i])
        w=np.subtract(w,a*q[:,1])
        b=la.norm(w)
        q[:,i+1]=w/b
       
    HQPB=q[:,2:k+2]
    print("HQPB este:",HQPB)  
    
    
    proiectare=np.dot(B.T,HQPB)
    print(proiectare)
    
    
    PR_PozaCaut=np.dot(PozaCaut,HQPB)
    print(PR_PozaCaut)



def cautareLA():
    configurareA()

    
    calePozaCautata=r'F:\zSCOALA\acs folder\proiect\att_faces\s1\10.pgm'
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVect=pozaCautata.reshape(-1,)

    plt.imshow(pozaCautataVect.reshape(112,92), cmap='gray',vmin=0, vmax=255) 
    plt.show()
    
    LA(pozaCautataVect)


cautareLA()
