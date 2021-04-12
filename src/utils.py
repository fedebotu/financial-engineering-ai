import torch; import torch.nn as nn
from torch import sin, pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential
import twint

# Data scraping
# Follow instructions here https://github.com/twintproject/twint/issues/1165
# and here https://github.com/twintproject/twint/issues/960
import twint # for collecting twitter data
import nest_asyncio
nest_asyncio.apply()

def get_tweets(start_date, end_date, company_name, company_ticker, lang='en', hide_output=True):
    c = twint.Config()
    c.Search = company_name, company_ticker
    c.Since = start_date
    c.Until = end_date
    c.Store_csv = True
    c.Lang = lang
    c.Count = True
    c.Hide_output = hide_output
    c.Format = "id: {id} | date: {date} | tweet: {tweet} | retweets_count: {retweets}"
    c.Custom['tweet'] = ['id', 'date', 'tweet', 'retweets_count']
    c.Output = 'data/' + company_name + '_data.csv'
    twint.run.Search(c)

# Pull request merged in https://github.com/EdwardDixon/snake 
# From https://github.com/Juju-botu/snake
class Snake(nn.Module):
    '''         
    Implementation of the serpentine-like sine-based periodic activation function
    
    .. math::
         Snake_a := x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        
    Parameters:
        - a - trainable parameter
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, a=None, trainable=True):
        '''
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model. 
        '''
        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        # Initialize `a`
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # Set the training to true

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        '''
        return  x + (1.0/self.a) * pow(sin(x * self.a), 2)