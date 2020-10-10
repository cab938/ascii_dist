# See https://asc-paint.glitch.me/ as a way to create fancy ANSI curves

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from zipfile import ZipFile
from IPython.core.magic import register_cell_magic, needs_local_scope

@register_cell_magic
@needs_local_scope
def ascii_dist(line: str, cell: str, local_ns: dict) -> None:
    r"""A cell magic to generate a probability distribution from an ASCII (really, unicode) drawing.
    The text in the cell represents a visual of the distribution which is set as the `dist` variable
    in the callers local scope.
    
    For instance, after a cell like this is run:
    
    %%ascii_dist
       _____
      /     \
     /       \
    /         \

    One can sample values from a (roughly normal) distribution by invoking `dist.sample()`.
    
    More unique distributions which might be more difficult to describe programmatically can be
    generated easily, such as this one:
    
    \
     \
      \_
        \_/\/\
    
    To visualize the continuous distribution from which this is formed call `dist.image()`
    
    Args:
        line: unused parameters (reserved for future use)
        cell: a unicode visual distribution
        local_ns: the callers local namespace where `dist` will be inserted
    
    Returns:
        None
    """
    distribution=ASCIIDistribution(cell, local_ns)
    local_ns['dist']=distribution
    return

class ASCIIDistribution:
    """Represents a probability distribution based on an ASCII diagram"""
        
    def __init__(self, ascii_data, local_ns, frequency='1S'):
        self.df=None
        
        # TODO: figure out how big the canvas should be from the size of the text entered
        img=Image.new("L", (500,250), color=255)

        # TODO: consider making the font choice an option, and choose size based on canvas descision
        with ZipFile('Source_Code_Pro.zip', 'r') as font_zip:
            with font_zip.open('SourceCodePro-Medium.ttf') as ttf_file:    
                scp_font=ImageFont.truetype(ttf_file,size=48)

        # Create drawing object and render our text
        dr=ImageDraw.Draw(img)
        dr=dr.text((0,0), ascii_data, font=scp_font)
        
        # render image to screen
        #display(img)

        WHITE_PIXEL_VALUE=255

        # calculate the values along the x axis of the number of white pixels under the curve or set
        # the value to None if there is no curve present at a given location
        pixels=[]
        for x in range(img.size[0]):
            num_white_pixels=0
            for y in range(img.size[1]):
                if img.getpixel((x,img.size[1]-1-y)) != WHITE_PIXEL_VALUE:
                    break
                num_white_pixels+=1
            if num_white_pixels==img.size[1]:
                pixels.append(None)
            else:
                pixels.append(num_white_pixels)

        # convert list to pandas to clean up distribution
        pixels=pd.Series(pixels)
        # trim the start and end of the pixel series from nans
        pixels=pixels[~pixels.isna()]
        # add back in gaps as nans by resampling to a secon frequency
        pixels.index=pd.to_datetime(pixels.index,unit='s')
        pixels=pixels.resample(frequency).asfreq()

        # interpolate to make distribution continuous
        pixels=pixels.interpolate(limit_direction="both")

        # smooth data with rolling window
        # TODO: Need to determine the size of the window based on size of the image/data
        pixels=pixels.rolling(25,min_periods=1,center=True).mean()

        # TODO: the date time index isn't needed any more so we should drop that back to integers?
        # TODO: we need to generate the probability of a given point being chosen and put that in df["prob"]
        # TODO: we need to generate the cummulative probability of a given point and put that in df["cum_prob"]
        # TODO: then we pull out np.random.uniform() and find the closest point in df["cum_prob"] to that
        # what's a good way to do this? lame way: np.min(np.abs(df["cum_prob"]-rando_num)) ??? 
        # Then we want to pull out that number's index. hrm.

        df=pd.DataFrame(pixels,columns=["datapoints"])
        df["prob"]=df["datapoints"]/df["datapoints"].sum()
        df["cumprob"]=df["prob"].cumsum()

        self.df=df

    def sample(self, size: int = 1) -> np.ndarray:
        """Pull a sample bounded by 0.0 and 1.0 from the distribution.
        Args:
            size: The number of samples to return.
        
        Returns:
            An ndarray of the samples drawn.
        """
        # initialize the return value array
        return_values=np.empty(size)
        
        # TODO: A loop is lame, there must be a straight forward way to broadcast this, no?
        for i in range(0,size):
            target=np.random.uniform()
            options=np.abs(self.df["cumprob"]-target)
            best_match=(options[options==np.min(options)].index[0]-options.index[0]).total_seconds()/len(options)
            return_values[i]=best_match
        
        return return_values
    
    def image(self) -> None:
        """Shows an image of the ASCII text converted into a smoothed continuous distribution for sampling.
        See `plot` for generating a simulation-based rendering of this distribution.
        """
        display(self.df["datapoints"].plot())
    
    def plot(self, n: int = 1000) -> None:
        """Samples from the distribution `n` times and plots the results. See `image` for an image of the
        distribution which does not require sampling.
        Args:
            n: Number of times to sample from the distribution
        
        Returns:
            None
        """
        display(pd.DataFrame(self.sample(n)).hist(bins=100))