from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.special import erf
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class knn_distinguish():
    def __init__(self, data: pd.DataFrame, predicted_class:list) -> None:
        '''
        data->the combined data, the whole data
        data_origin->the original data, aka the test data
        predicted_class->a bunch of class from data, class name, we assume it's a list
        '''
        # print(len(data))
        self.store_LSE = {}
        self.arctan_popt = {}
        self.logistic_popt = {}
        self.tanh_popt = {}
        self.arc_popt = {}
        self.gd_popt = {}
        self.ERF_popt = {}
        self.algebra_popt = {}
        self.Gompertz_popt = {}
        
        #store for edge case:
        self.arctan_min = {}
        self.logistic_min = {}
        self.tanh_min = {}
        self.arc_min = {}
        self.gd_min = {}
        self.ERF_min = {}
        self.algebra_min = {}
        self.Gompertz_min = {}
        
        self.data = data
        # self.data_origin = data_origin
        self.predicted_class = predicted_class
        self.maxDis_train = 0
        self.df_comb1 = None

    def combine_data(self, data, classes):
        '''
        split the data according to the classes
        '''
        df = pd.DataFrame(data)
        for index, name in enumerate(classes):
            df.loc[df.iloc[:, 0] == name, 'class'] = name
        df = df.drop(df.columns[0], axis=1).drop_duplicates()
        return df
    
    def data_process(self, dataframe, class_name):
        a = dataframe[dataframe['class']==class_name].drop("class",axis=1).astype(float).to_numpy()
        return a

    def data_distance(self, data):
        '''
        calculating empirical data's shortest(NN) distance 
        real data is high-dimensional data points
        '''
        if len(data) == 1:
            return np.array(data)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
        distances, _ = nbrs.kneighbors(data)
        shortest_distances = distances[:, 1]
        if not np.any(shortest_distances):
            return np.array([0] * len(data))
        return shortest_distances
    
    def empirical_CDF(self, data):
        '''
        return x,y data of CDF 
        '''   
        sort_data = np.sort(data)
        self.maxDis_train =np.max(sort_data)
        if sort_data.ndim == 1:
            x = np.concatenate(([0],sort_data))
        else:
            x = np.concatenate(([0], sort_data.reshape(-1)))
        y = np.zeros((x.shape))
        for i in range(len(x)):
            # if i == 0:
            #     print((len(x)-i-0.5)/len(x))
            y[i] = (len(x)-i-0.5)/len(x)
        print(y)
        return x,y
    
    def auto_curve_fit(self, data_NN, x, y, x_scale_factor, func, s, p_control=None):
        '''
        data_NN: array empirical data_distance for calculating median
        x,y: from CDF
        s: sigma in curve_fit(), for weighting
        '''
        if p_control == "Gompertz":
            p0 = [1,1]
        elif p_control == "Weight":
            p0 = [np.median(data_NN)/x_scale_factor,1,0.5]
        else:
            p0 = [np.median(data_NN)/x_scale_factor,1] # this is initial guess for sigmoid parameters
        
        if not np.any(x):
            try:
                popt, _ = curve_fit(f=func, xdata=x, ydata=y, p0=p0,method='lm')
            except TypeError:
                popt = np.zeros(5)
        else:
            try:
                popt, _ = curve_fit(f=func, xdata=x/x_scale_factor, ydata=y, p0=p0,method='lm')
            except TypeError:
                popt = np.zeros(5)
        return popt
    
    def data_binning(self, data):
        x = np.sort(data) 
        N = len(x)                   # e.g N = 500, sqrt(500)=22.3
        lower = int(np.floor(np.sqrt(N))) # 22
        upper = int(np.ceil(np.sqrt(N)))  # 23 as total #of bin
        
        if lower*upper >= N:
            small_bin_num = int(lower*upper - N)  # 22*23 - 500 = 6
            small_bin_size = int(lower - 1)  # 21
            large_bin_size = lower
        else: # HGG -> sqrt(252) = 15.8
            small_bin_num = int(upper**2 - N) # 16*16-252 =4
            small_bin_size = lower  # 15
            large_bin_size = upper
        
        large_bin_num = int(upper - small_bin_num) # 23-6 = 17

        # small_bin_size*small_bin_num + lower*large_bin_num = N

        bin_count = [large_bin_size]*large_bin_num + [small_bin_size]*small_bin_num  # [22..*17, 21..*6,]
        #print("items in each bin: ", bin_count)
        binned_data = []
        i = 0
        for count in bin_count:
            binned_data.append(np.mean(x[i:i+count]))
            i += count
        
        return binned_data


    def binning_xy(self, binned_data):
        x = np.concatenate(([0],binned_data))
        y = np.zeros((x.shape))
        for i in range(len(x)):
            y[i] = (len(x)-i-0.5)/len(x)
        return x,y
    
    
    def sigmoids_for_class(self, data, name, factor, func_list, binning=False, plot=False): #removed color list here
        if binning:
            x,y = self.binning_xy(self.data_binning(data))
        else:
            x,y = self.empirical_CDF(data)
        if plot:
            color_list = ['g','r','c','m','y','k','brown','gray']
            f,ax = plt.subplots(1,2,figsize=(16,6))
            ax[0].set_title('1-y(p_value) of '+ str(name))
            ax[0].set_yscale('log')
            ax[0].scatter(x,1-y, color='b',s=10)

            ax[1].set_title('y of '+name)
            ax[1].scatter(x,y, color='b',s=10)
        res = []
        #print("for ", name," :")
        for i in range(len(func_list)):
            try:
                if i == 7:
                    p = self.auto_curve_fit(data,x,y,factor,func_list[i],s=y,p_control="Gompertz")
                elif i == 6:
                    p = self.auto_curve_fit(data,x,y,factor,func_list[i],s=y,p_control="Weight")
                else:
                    p = self.auto_curve_fit(data,x,y,factor,func_list[i],s=y)
            except RuntimeError:
                print("error in ",str(func_list[i])[9:-22])
                continue
            smoothing_term = 1e-10
            if np.array_equal(p, np.zeros(5)):
                y2 = 0
                y_true = 1-y + smoothing_term
                y_true_filtered = y_true[y_true > 0]
                error = np.sum(np.square(np.log(y_true_filtered + smoothing_term)))
            else:
                y2 = func_list[i](x/factor, *p)
                y_pred = y2
                y_true = y
                y_pred_filtered = y_pred[y_pred > 0]
                y_true_filtered = y_true[y_pred > 0]
                error = np.sum(np.square(np.log(y_pred_filtered + smoothing_term) - np.log(y_true_filtered + smoothing_term)))

            if func_list[i] == self.arctan_GD:
                self.arctan_popt[f"{name}"] = p
                self.arctan_min[f"{name}"] = np.min(y2)
                self.store_LSE["arctan_gd"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='arctan_GD')
                    ax[1].plot(x, y2, color=color_list[i], label='arctan_GD')
                # print(np.min(y2))
                res.append([func_list[i], *p])
            if func_list[i] == self.logistic:
                self.logistic_popt[f"{name}"] = p
                self.logistic_min[f"{name}"] = np.min(y2)
                self.store_LSE["logistic"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='logistic')
                    ax[1].plot(x, y2, color=color_list[i], label='logistic')
                res.append([func_list[i], *p])
            if func_list[i] == self.tanh:
                self.tanh_popt[f"{name}"] = p
                self.tanh_min[f"{name}"] = np.min(y2)
                self.store_LSE["Hyperbolic tangent"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='Hyperbolic tangent')
                    ax[1].plot(x, y2, color=color_list[i], label='Hyperbolic tangent')
                res.append([func_list[i], *p])
            if func_list[i] == self.arctan:
                self.arc_popt[f"{name}"] = p
                self.arc_min[f"{name}"] = np.min(y2)
                self.store_LSE["arctan"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='arctan')
                    ax[1].plot(x, y2, color=color_list[i], label='arctan')
                res.append([func_list[i], *p])
            if func_list[i] == self.GD:
                self.gd_popt[f"{name}"] = p
                self.gd_min[f"{name}"] = np.min(y2)
                self.store_LSE["Gudermannian"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='Gudermannian')
                    ax[1].plot(x, y2, color=color_list[i], label='Gudermannian')
                res.append([func_list[i], *p])
            if func_list[i] == self.ERF:
                self.ERF_popt[f"{name}"] = p
                self.ERF_min[f"{name}"] = np.min(y2)
                self.store_LSE["ERF"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='ERF')
                    ax[1].plot(x, y2, color=color_list[i], label='ERF')
                res.append([func_list[i], *p])
            if func_list[i] == self.algebra:
                self.algebra_popt[f"{name}"] = p
                self.algebra_min[f"{name}"] = np.min(y2)
                self.store_LSE["algebraic"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='algebraic')
                    ax[1].plot(x, y2, color=color_list[i], label='algebraic')
                res.append([func_list[i], *p])
            if func_list[i] == self.Gompertz:
                self.Gompertz_popt[f"{name}"] = p
                self.Gompertz_min[f"{name}"] = np.min(y2)
                self.store_LSE["Gompertz"] = error
                if plot:
                    ax[0].plot(x, 1-y2, color=color_list[i], label='Gompertz')
                    ax[1].plot(x, y2, color=color_list[i], label='Gompertz')
                res.append([func_list[i], *p])
        if plot:
            ax[0].legend(loc='lower left')
            ax[1].legend(loc='lower left')
            plt.show()
        return res


    def logistic(self, x,x0, k):
        m = (1/ (1 + np.exp(-k*(x-x0))))     
        return np.clip(m, 0, 1)

    def tanh(self, x, x0, k): 
        m = (1+np.tanh(k*(x-x0)))/2    
        return np.clip(m, 0, 1)

    def arctan(self, x, x0, k):
        m = (1+(2/np.pi)*np.arctan(k*(x-x0)))/2
        return np.clip(m, 0, 1)

    def GD(self, x, x0, k):
        m = (1+(4/np.pi)*np.arctan(np.tanh(k*(x-x0))))/2
        return np.clip(m, 0, 1)

    def ERF(self, x, x0, k):
        m = (1+erf(k*(x-x0)))/2
        return np.clip(m, 0, 1)

    def algebra(self, x, x0, k):
        abs_x = abs(x)
        denominator = (1 + abs_x ** k) ** (1/k)
        m = (1 + x / denominator) / 2
        # m = (1+x/((1+abs(x)**k)**(1/k)))/2
        if np.any(denominator == 0):
            m[denominator == 0] = 0
        return np.clip(m, 0, 1)

    def arctan_GD(self, x,x0,k, w):
        m = w*self.GD(x,x0,k)+(1-w)*self.arctan(x,x0,k)
        return np.clip(m, 0, 1)

    def Gompertz(self, x,b,c):
        m = np.e**(-np.e**(b-c*x))
        return np.clip(m, 0, 1)
        
    def build_AM(self, x, y):
        """Function to build the A matrix in parallel.

        Keyword arguments:
        x -- the input data numpy array in the form nxm (samplesxfeatures).
        y -- the numpy array that represents the classes for each sample.
        """
        def build_cm(feat):
            cm = []
            for lab in np.unique(y):
                x_fit = x[np.where(y == lab)[0], feat].reshape(-1, 1)
                params = {'bandwidth': np.linspace(0.01, 1, 30)}
                grid = GridSearchCV(KernelDensity(), params, cv=5)
                grid.fit(x_fit)
                kde = grid.best_estimator_
                a = np.exp(kde.score_samples(x_fit))
                cm.extend(a.tolist())
            return np.array(cm, dtype=object).flatten()

        bm = Parallel(n_jobs=-1)(delayed(build_cm)(feat) for feat in range(x.shape[1]))
        bm = np.array(bm).T
        return bm



    def distanceValue(self, test_data: pd.DataFrame, sigmoid_function, sigmoid_popt, df_combine, edge_dict):

        nearest_NN = []
        sigmoid_val = []

        for target in sigmoid_popt:
            distances = pairwise_distances(test_data.to_numpy(), df_combine[df_combine['class'] == f'{target}'].drop(['class'], axis = 1).to_numpy(), metric='euclidean')
            nearDis = np.min(distances)
            nearest_NN.append(nearDis)
            # if sigmoid_function(nearDis, *sigmoid_popt[f'{target}']) == 0 and edge_dict != {}:
            #     nearDis = self.maxDis_train
            #     sigmoid_val.append(edge_dict[f'{target}'])
            # else:
            sigmoid_val.append(sigmoid_function(nearDis, *sigmoid_popt[f'{target}']))
        return [sigmoid_val, nearest_NN]

    def helper_plot_curve(self, final_list, nearest_nn, sigmoid_val):
        popt_list = [v for v in final_list[1].values()]
        for i in range(len(popt_list)):
            plt.title(self.predicted_class[i])
            x = np.linspace(0, nearest_nn[i]+1, 100)
            y = final_list[0](x, *popt_list[i])
            plt.plot(x, y, label="train sigmoid")
            print("nearDis: ", nearest_nn)
            print("sigmoid Val: ", sigmoid_val)
            plt.scatter(nearest_nn[i], sigmoid_val[i], label="test point", c='red')
            plt.legend()
            plt.show()

    def fit(self, if_binning=False, plot=False):
        df_comb = self.combine_data(self.data, self.predicted_class)
        self.df_comb1 = df_comb
        final_list = []
        df_class = df_comb['class']
        mpVal = {}
        for i, v in enumerate(self.predicted_class):
            mpVal[v] = i+1
        X = df_comb.iloc[:, :-1].values
        y = df_comb.iloc[:, -1].values
        df_comb = self.build_AM(X, y)
        df_comb = pd.DataFrame(df_comb)
        df_comb = pd.DataFrame(df_comb, columns=list(df_comb.columns) + ['class'])
        df_comb['class'] = df_class.reset_index(drop = True)
        functions = [ self.logistic, self.tanh, self.arctan, self.GD, self.ERF, self.algebra, self.arctan_GD, self.Gompertz]
        for i in range(len(self.predicted_class)):
            processed_data = self.data_distance(self.data_process(df_comb, self.predicted_class[i]))
            # print(np.max(processed_data))
            #[func, *p]
            final_dict = self.sigmoids_for_class(processed_data, self.predicted_class[i], np.mean(processed_data), functions, binning=if_binning, plot=plot)
            # print(final_dict)
            final_list.append(final_dict)
            # print(final_dict)
        # print(self.store_MSE)
        sorted_sigmoid = sorted(self.store_LSE.items(), key=lambda x:x[1])
        #print(sorted_sigmoid)
        sig_function = sorted_sigmoid[0][0]
        if sig_function == "arctan_gd":
            return self.arctan_GD, self.arctan_popt, self.arctan_min
        if sig_function == "ERF":
            return self.ERF, self.ERF_popt, self.ERF_min
        if sig_function == "arctan":
            return self.arctan, self.arc_popt, self.arc_min
        if sig_function == "logistic":
            return self.logistic, self.logistic_popt, self.logistic_min
        if sig_function == "Hyperbolic tangent":
            return self.tanh, self.tanh_popt, self.logistic_min
        if sig_function == "Gudermannian":
            return self.GD, self.gd_popt, self.gd_min
        if sig_function == "algebraic":
            return self.algebra, self.algebra_popt, self.algebra_min
        if sig_function == "Gompertz":
            return self.Gompertz, self.Gompertz_popt, self.Gompertz_min
        
    def predict(self, test_data: pd.DataFrame, sig_function, sig_popt: np.ndarray, edgecase=None):
        val = []
        if edgecase != None:
            edge_dict = edgecase
        else:
            edge_dict = {}
        
        val = self.distanceValue(test_data, sig_function, sig_popt, self.df_comb1, edge_dict)[0]
        nearest_NN = self.distanceValue(test_data, sig_function, sig_popt, self.df_comb1, edge_dict)[1]
        print(f"nearest NN" + str(nearest_NN))

        print(f"sigmoid value" + str(val))

        print(str(sig_function))
        final_list = [sig_function, sig_popt]
        print(sig_popt)
        self.helper_plot_curve(final_list, nearest_NN, val)
        
        return val
    
    def accurcy_score(self):
        return