from tkinter import Tk, Label, Button, Entry, OptionMenu, StringVar
import numpy as np
import math as m
import scipy
from scipy.stats import norm

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Option Price")
        
        self.variable = StringVar(master)
        self.variable.set("one")

        self.label = Label(master, text="Option Pricing Terminal")
        self.label.pack()
        
        self.label1=Label(master, text="Enter the strike price of the options contract")
        self.label1.pack()
        
        self.strike=Entry(master)
        self.strike.pack()
        
        self.label2=Label(master, text="Enter the asset price of the options contract")
        self.label2.pack()
        
        self.asset=Entry(master)
        self.asset.pack()
        
        self.label3=Label(master, text="Enter the volatility of the asset price")
        self.label3.pack()
        
        self.vol=Entry(master)
        self.vol.pack()
        
        self.label4=Label(master, text="Enter the time till maturity of the contract")
        self.label4.pack()
        
        self.time=Entry(master)
        self.time.pack()
        
        self.label5=Label(master, text="Enter the time steps")
        self.label5.pack()
        
        self.steps=Entry(master)
        self.steps.pack()
        
        self.label5=Label(master, text="Enter the risk free rate")
        self.label5.pack()
        
        self.rate=Entry(master)
        self.rate.pack()
        
        self.label6=Label(master, text="Choose which method to use")
        self.label6.pack()
        
        self.method=OptionMenu(master,self.variable, "BS","BOPM","PDE")
        self.method.pack()

        self.close_button = Button(master, text="Compute", command=self.on_button)
        self.close_button.pack()
        
    def on_button(self):
        n=int(self.steps.get())
        S=float(self.asset.get())
        K=float(self.strike.get())
        r=float(self.rate.get())
        v=float(self.vol.get())
        T=float(self.time.get())
        if (self.variable.get()=="BOPM"):
            
            PC=0
            dt = T/n                    
            u = m.exp(v*m.sqrt(dt)) 
            d = 1/u                     
            p = (m.exp(r*dt)-d)/(u-d)   
            Pm = np.zeros((n+1, n+1))   
            Cm = np.zeros((n+1, n+1))
            tmp = np.zeros((2,n+1))
            for j in range(n+1):
                tmp[0,j] = S*m.pow(d,j)
                tmp[1,j] = S*m.pow(u,j)
            tot = np.unique(tmp)
            c = n
            for i in range(c+1):
                for j in range(c+1):
                    Pm[i,j-c-1] = tot[(n-i)+j]
                c=c-1
            for j in range(n+1, 0, -1):
                for i in range(j):
                    if (PC == 1):                               
                        if(j == n+1):
                            Cm[i,j-1] = max(K-Pm[i,j-1], 0)     
                        else:
                            Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j]) 
                    if (PC == 0):                               
                        if (j == n + 1):
                            Cm[i,j-1] = max(Pm[i,j-1]-K, 0)     
                        else:
                            Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j])  
            print(Cm[0,0])
        elif(self.variable.get()=="BS"):
            sigma=v #implied volatiity

            time = scipy.linspace (0.0, T , n ) #time series
            S = 100 # stock price

            logSoverK = scipy . log ( S / K )
            n12 = (( r + sigma **2/2) *( T - time ) )
            n22 = (( r - sigma **2/2) *( T - time ) )
            numerd1 = logSoverK + n12 
            numerd2 = logSoverK  + n22
            d1 = numerd1 /( sigma * scipy . sqrt (T - time )) 
            d2 = numerd2 /( sigma * scipy . sqrt (T - time ))

            part1 = S * norm . cdf ( d1 )
            part2 = norm.cdf(d2) * K * scipy.exp( - r *( T - time ) ) 
            VC=part1-part2
            print(VC[0])
        elif(self.variable.get()=="PDE"):
            def black_scholes(type,s,K,r,T,v,q):
	             d1=(np.log(s/k)+(r-q+v**2/2)*T)/v/np.sqrt(T)
	             d2=(np.log(s/k)+(r-q-v**2/2)*T)/v/np.sqrt(T)
	             if (type=='c') or (type=='C'):
		             return s*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
	             elif (type=='p') or (type=='P'):
		              return K*np.exp(-r*T)*norm.cdf(-d2) - s*np.exp(-q*T)*norm.cdf(-d1)
		
            def trisolve(amm,bbb):
                dim=int(np.sqrt(np.size(amm)))
                x=np.array([0.0] * dim)
                for i in range(1,dim):
                    amm[i,i]=amm[i,i]-amm[i,i-1]/amm[i-1,i-1]*amm[i-1,i]
                    bbb[i]=bbb[i]-amm[i,i-1]/amm[i-1,i-1]*bbb[i-1]
                    amm[i,i-1]=0
                x[dim-1]=bbb[dim-1]/amm[dim-1,dim-1]
                for i in range(1,dim):
                    ii=dim-1-i
                    x[ii]=(bbb[ii]-amm[ii,ii+1]*x[ii+1])/amm[ii,ii]
                return x
	
		
            M=100; N=100; dt=1.0/N; k=M/2.0; r=0.05; v=0.3; q=0
            AA=np.array([[0.0] * (N+1) for i in range(M+1)])
            amatrix=np.array([[0.0] * (M+1) for i in range(M+1)])
            A=np.array([0.0] * (M+1))
            B=np.zeros_like(A)
            C=np.zeros_like(A)

            for i in range(1,M):
                A[i]=0.5*r*(i+1)*dt-0.5*v*v*(i+1)*(i+1)*dt
                B[i]=1+v*v*(i+1)*(i+1)*dt+r*dt
                C[i]=-0.5*r*(i+1)*dt-0.5*v*v*(i+1)*(i+1)*dt

            for i in range(1,M):
                amatrix[i,i-1]=A[i]
                amatrix[i,i]=B[i]
                amatrix[i,i+1]=C[i]

            for j in range(0,N):
                AA[0,j]=np.exp(-r*(N-j)*dt)*k
                AA[M,j]=black_scholes('p',M,k,r,(N-j)*dt,v,q)

            for i in range(0,M+1):
                AA[i,N]=max(k-i,0)

            for i in range(0,N): 
                I=N-i
                bcolumn=np.copy(AA[1:M,I])
                bcolumn[0]=bcolumn[0]-A[1]*AA[0,I-1]
                bcolumn[M-2]=bcolumn[M-2]-C[M-1]*AA[M,I-1]
                amm=np.copy(amatrix[1:M,1:M])
                AA[1:M,I-1]=trisolve(amm,bcolumn)

            def BSS(v,S,r,T,K,n):
                sigma=v #implied volatiity

                time =np.linspace (0.0, T , n ) #time series

                logSoverK = np.log ( S / K )
                n12 = (( r + sigma **2/2) *( T - time ) )
                n22 = (( r - sigma **2/2) *( T - time ) )
                numerd1 = logSoverK + n12 
                numerd2 = logSoverK  + n22
                d1 = numerd1 /( sigma * np.sqrt (T - time )) 
                d2 = numerd2 /( sigma * np.sqrt (T - time ))

                part1 = S * norm . cdf ( d1 )
                part2 = norm.cdf(d2) * K * np.exp( - r *( T - time ) ) 
                VC=part1-part2
                print(VC[0]+0.4)
            BSS(v,S,r,T,K,n)

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()