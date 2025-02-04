# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
    '''
    This class implements Recurrent Neural Networks.
    
    You should implement code in the following functions:
        predict				->	predict an output sequence for a given input sequence
        acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.
        
        DO NOT CHANGE THIS
        
        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        with is_param():
            self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
            self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

        # matrices to accumulate weight updates
        with is_delta():
            self.deltaU = np.zeros_like(self.U)
            self.deltaV = np.zeros_like(self.V)
            self.deltaW = np.zeros_like(self.W)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        
        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word
        
        '''
        
        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )

        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            #This should be vocab
            vector = np.zeros((self.vocab_size,1)) #ensure this is a column vector
            vector[x[t],0] = 1
            #Ensure this returns a column vector of probabilities (i.e. it broadcasts)
            s_current_vector = sigmoid(self.V @ vector + self.U @ s[t-1, :].reshape(-1, 1))
            
            net_out = self.W @ s_current_vector

            y[t,:] = softmax(net_out).reshape(1,-1)
            s[t,:] = s_current_vector.T




        return y, s
    
    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        d	list of words, as indices, e.g.: [4, 2, 3]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''

        for t in reversed(range(len(x))):
            #Need to use t+1 (because this will go from t = n-1 to t=0)

            # Create a one-hot encoded vector for the current target word
            current_d = np.zeros((self.out_vocab_size, 1))
            current_x = np.zeros((self.vocab_size, 1))
            current_d[d[t]] = 1
            current_x[x[t]] = 1
            
            # Reshape the predicted output for the current time step to a column vector
            y_t = np.array(y[t]).reshape(-1, 1)

            # Calculate the output layer error (difference between target and predicted output)
            delta_t_out = (current_d - y_t)
            print(delta_t_out.shape)

            # Accumulate the gradient for W using the outer product of delta_t_out and the hidden state
            outer_product_W_t = delta_t_out @ (np.array(s[t]).reshape(1, -1))
            self.deltaW += outer_product_W_t

            # Calculate the derivative of the activation function (sigmoid) for the hidden state
            f_dash_net_in = (s[t] * (1 - s[t])).reshape(-1,1)
            print(f_dash_net_in.shape)
            
            # Calculate the error for the hidden layer
            delta_t_in = (self.W.T @ delta_t_out) * f_dash_net_in
            print(delta_t_in.shape)
            # print(outer_product_V_t.shape)

            # Accumulate the gradient for V using the outer product of delta_t_in and the input vector
            outer_product_V_t = delta_t_in @ (current_x.reshape(1, -1))
            
            self.deltaV += outer_product_V_t

            # Accumulate the gradient for U using the outer product of delta_t_in and the previous hidden state
            outer_product_U_t = delta_t_in @ (np.array(s[t-1]).reshape(1, -1))
            self.deltaU += outer_product_U_t


            
            

    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''

        #QUESTION: Don't think anything changes here except we don't look through time?
        assert len(d) == 1

        #Set current time (zero indexed)
        t = len(x)-1

        #For the number prediction task, we expect self.out_vocab_size = 2
        current_d = np.zeros((self.out_vocab_size, 1))

        #WARNING - TO UPDATE: need to make sure current_x is set to vobab_size but current_d is set to output vocab size (Not necessarily the same if doing prediction!)
        current_x = np.zeros((self.vocab_size, 1))
        
        #Be careful here 
        current_d[d[0]] = 1
        current_x[x[t]] = 1

        #So our predict function doesn't change but we just care about the last one?
        #I.e. we still predict a label for each word
        
        # Reshape the predicted output for the current time step to a column vector
        y_t = np.array(y[t]).reshape(-1, 1)

        # Calculate the output layer error (difference between target and predicted output)
        delta_t_out = (current_d - y_t)

        # Accumulate the gradient for W using the outer product of delta_t_out and the hidden state
        outer_product_W_t = delta_t_out @ (np.array(s[t]).reshape(1, -1))
        self.deltaW += outer_product_W_t

        # Calculate the derivative of the activation function (sigmoid) for the hidden state
        f_dash_net_in = (s[t] * (1 - s[t])).reshape(-1,1)
        
        # Calculate the error for the hidden layer
        delta_t_in = (self.W.T @ delta_t_out) * f_dash_net_in


        # Accumulate the gradient for V using the outer product of delta_t_in and the input vector
        outer_product_V_t = delta_t_in @ (current_x.reshape(1, -1))
        
        self.deltaV += outer_product_V_t

        # Accumulate the gradient for U using the outer product of delta_t_in and the previous hidden state
        outer_product_U_t = delta_t_in @ (np.array(s[t-1]).reshape(1, -1))
        self.deltaU += outer_product_U_t



        
    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''

        for t in reversed(range(len(x))):
            #Need to use t+1 (because this will go from t = n-1 to t=0)

            # Create a one-hot encoded vector for the current target word
            current_d = np.zeros((self.out_vocab_size, 1))
            current_x = np.zeros((self.vocab_size, 1))
            current_d[d[t]] = 1
            current_x[x[t]] = 1
            
            # Reshape the predicted output for the current time step to a column vector
            y_t = np.array(y[t]).reshape(-1, 1)

            # Calculate the output layer error (difference between target and predicted output)
            delta_t_out = (current_d - y_t)

            # Accumulate the gradient for W using the outer product of delta_t_out and the hidden state
            outer_product_W_t = delta_t_out @ (np.array(s[t]).reshape(1, -1))
            self.deltaW += outer_product_W_t

            # Calculate the derivative of the activation function (sigmoid) for the hidden state
            f_dash_net_in_t = (s[t] * (1 - s[t])).reshape(-1,1)
            
            # Calculate the error for the hidden layer
            delta_t_in = (self.W.T @ delta_t_out) * f_dash_net_in_t

            # Accumulate the gradient for V using the outer product of delta_t_in and the input vector
            outer_product_V_t = delta_t_in @ (np.array(current_x).reshape(1, -1))
            self.deltaV += outer_product_V_t

            # Accumulate the gradient for U using the outer product of delta_t_in and the previous hidden state
            outer_product_U_t = delta_t_in @ (np.array(s[t-1]).reshape(1, -1))
            self.deltaU += outer_product_U_t

            delta_in_t_minus_tau_plus_1 = delta_t_in
            tau = 1
            while (t-tau >=0) and tau <= steps:
                f_dash_net_in_t_minus_tau = (s[t-tau] * (1 - s[t-tau])).reshape(-1,1)
                delta_in_t_minus_tau = (self.U).T @ delta_in_t_minus_tau_plus_1 * f_dash_net_in_t_minus_tau
                x_t_minus_tau = np.zeros((self.vocab_size, 1))
                x_t_minus_tau[x[t-tau]] = 1
                self.deltaV += delta_in_t_minus_tau @ (x_t_minus_tau.reshape(1, -1))
                self.deltaU += delta_in_t_minus_tau @ ((np.array(s[t-tau-1]).reshape(1, -1)))

    
                #Set current delta to previous delta and loop
                delta_in_t_minus_tau_plus_1 = delta_in_t_minus_tau
                tau +=1 



    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''

        #I think the only difference is you just remove the reversed function

        t = len(x) - 1
        # Create a one-hot encoded vector for the current target word
        current_d = np.zeros((self.out_vocab_size, 1))
        #Make sure current x is vocab size, not output vocab size
        current_x = np.zeros((self.vocab_size, 1))
        current_d[d[0]] = 1
        current_x[x[t]] = 1
        
        # Reshape the predicted output for the current time step to a column vector
        y_t = np.array(y[t]).reshape(-1, 1)

        # Calculate the output layer error (difference between target and predicted output)
        delta_t_out = (current_d - y_t)

        # Accumulate the gradient for W using the outer product of delta_t_out and the hidden state
        outer_product_W_t = delta_t_out @ (np.array(s[t]).reshape(1, -1))
        self.deltaW += outer_product_W_t

        # Calculate the derivative of the activation function (sigmoid) for the hidden state
        f_dash_net_in_t = (s[t] * (1 - s[t])).reshape(-1,1)
        
        # Calculate the error for the hidden layer
        delta_t_in = (self.W.T @ delta_t_out) * f_dash_net_in_t

        # Accumulate the gradient for V using the outer product of delta_t_in and the input vector
        outer_product_V_t = delta_t_in @ (np.array(current_x).reshape(1, -1))
        self.deltaV += outer_product_V_t

        # Accumulate the gradient for U using the outer product of delta_t_in and the previous hidden state
        outer_product_U_t = delta_t_in @ (np.array(s[t-1]).reshape(1, -1))
        self.deltaU += outer_product_U_t

        delta_in_t_minus_tau_plus_1 = delta_t_in
        tau = 1
        while (t-tau >=0) and tau <= steps:
            f_dash_net_in_t_minus_tau = (s[t-tau] * (1 - s[t-tau])).reshape(-1,1)
            delta_in_t_minus_tau = (self.U).T @ delta_in_t_minus_tau_plus_1 * f_dash_net_in_t_minus_tau
            x_t_minus_tau = np.zeros((self.vocab_size, 1))
            x_t_minus_tau[x[t-tau]] = 1
            self.deltaV += delta_in_t_minus_tau @ (x_t_minus_tau.reshape(1, -1))
            self.deltaU += delta_in_t_minus_tau @ ((np.array(s[t-tau-1]).reshape(1, -1)))


            #Set current delta to previous delta and loop
            delta_in_t_minus_tau_plus_1 = delta_in_t_minus_tau
            tau +=1 