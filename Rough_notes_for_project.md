Coding a complete tranformer from scratch

![alt text](image-4.png)


**WorkFlow ->**
1. Input Embedding
2. Positional Encoding
3. MultiHead Attention
4. Add and Normalization
5. FeedForward Network
6. Residual Network
7. Encoder
8. Decoder and projection layer
9. Building a transformer
10. Test our Transformer
11. Tokenizer
12. Loading dataset
13. Validation Loop
14. Training Loop
15. Conclusion



-------we did wwas impoting lib ---->

torch for tensor
nn for neural network
dataset from utils for dataset
dataloder for easily division in btach and iterabel 
random_split for 80/20

importing math for mathematics
import summaryWriter for the logs on terminal

hugging face we are just impoting mainly a tokenizer model 
atltogh we have tokenizer here too its work is to convert raw text to an id we will se it moving forward
we also imported whitespace of hugging face its work is like chunking
we have wordlevel as our model from tokenizer.model
and lso there is wordlevel trainer for traing the model  

path for finding path in computer from pathlib

warning for warning

tqdm for beaituful visual

also install datasets

thats all the lib needed 


now lets see each topic ->



1.----input embedding -----



Token-> my name is Himanshu
"My" "Name" "is" "Himanshu"

vocabliary->just unique work above there can be same words but in vocab w ewill have each word only one time
My,Name,is,Himanshu
now you can give here only them an indexing
My=0,Name=1,is=2,Himanshu=3
(Note-> no repeat )
now you can write them in number-> 1,2,3,4
now we write them in 2,3,1,4 you can read it from vocab and understand the sentence
now time to convert them to embeddings means

word to vec->
every single word will be converted to  number of her 512 vector remember it as d_model
example -> Animal = [0.2,0.1, 0.5.....] 
           humans = [0.2,0.12,0.49... ]
no one in world know what each of these vector point represent but we can say that if we draw the vector then we can see the close point with each other and far points away from each other 
now in that graph you can see word similar to each near each other and that have no same that word would be far from each other
you can study more wordtovec if you want to study it more in detail

context window ->
assume - A animal is dancing in the jungle .
let take context window 1 so it can oly look at one thing in that text before and after the the target word(ex: is ) either animal ,  dancing .

how do we make it we make like if i send "the" i will get "animal"
when i send "animal" i get "is " and "the"
this way we will make a model simple linear model of 512 hidden layer now htere weights will be the embedding here . 


code for input embedding ->
make sure you know constructor and oops in python for it .


class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int , vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

nn.embeddiing -> Think of nn.Embedding as a giant lookup table (a matrix) with vocab_size rows and d_model columns.

x is just a input tensor (a batch of word IDs)
self.embedding(x) do the lookup gives you the 
multiply by math.sqrt of d_model it is takne from research paper it is normally one to normalize so that some value donot get too large 

some ai generated expalnitation 

Step 1: Create the One InstanceBefore training begins, you initialize the layer once.Python# We create this ONCE. 
# It holds the giant lookup table for the whole vocabulary.
embed_layer = InputEmbeddings(d_model=512, vocab_size=30000)
Step 2: Tokenize the InputYour sentence is broken down into tokens, and the tokenizer converts them into integer IDs.Words: ["my", "name", "is", "himanshu"]IDs: [102, 450, 12, 8943]Step 3: The Forward Pass (Passing the Tensor)You wrap those 4 IDs into a single PyTorch tensor.In PyTorch, the mathematical shape of this input tensor is $(1, 4)$, which stands for:$1$ batch (one sentence)$4$ sequence length (four words)Pythonx = torch.tensor([[102, 450, 12, 8943]]) # Shape: (1, 4)
Now, you push that entire tensor through your single embed_layer instance at once:Pythonoutput = embed_layer(x) 
The Final Output ShapeThe embed_layer looks up the 512-dimensional vector for 102, the vector for 450, etc., and stacks them together.The mathematical shape of your output tensor is now $(1, 4, 512)$.$1$: The batch size (one sentence)$4$: The sequence length (four words)$512$: The embedding dimension ($d_{\text{model}}$)


2.---Positional Encoding ---

add image of till now and a foramula image for sure 

now we are sending all encoding aat one maybe computer is fast it can send randomly send embedding maybe it send 1st word at last so we need to tell model the position of each piece

just apply formula which is from paper
![image.png](attachment:c5bbf213-e133-4a65-9926-b3b52dff53d1.png)

![alt text](image-6.png)

here d_model = 512


* sin() for even
* cos() for odd

assume "The" word have embedding [0.2,0.3...] so apply sin on 0.2 and cos on 0.3 and continue for all 512 vector of the. 

why do we use this formula one hign it does is now value will be b/w -1 and 1 so normalization ndone too 

some change in formula to code it 

![alt text](image-7.png)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int ,dropout:float):

        super()__init__()

        self.d_model = d_model

        self.seq_len = seq_len

        self.dropout=nn.Dropout(dropout)



        pe = torch.zeros(seq_len,d_model)

        position = torch.arrange(0,seq_len,dtype=torch.float).unsqueeze(1) #[seq_len,1] unsequze add one dimension

        # div_term=torch.exp(torch.arrange(0,d_model,2).float() ) this was the code acc. to paper but that log beacuse for better acc to researcher down there

        div_term=torch.exp(torch.arrange(0,d_model,2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position * div_term)

        pe[:,1::2] = torch.cos(position * div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe) #we donot need to train them for that we add this line



    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
                
seq_len is max length of a sentence our model can handle
the above is the code till dropout normal thing while drop out is use d to prevent overfitting

torch.zeros() this thing created a canvas for us to work on
here assume seq len = 5000 and d_model =512 so it is 5000*512

torch.arrange just created a a list oof number from 0,4999 these are our word llik word0, wor1, --

.unsqueeze(1) just add a cloum so shae becomes (5000,1)


we needed to do this $$\text{Denominator} = 10000^{\frac{2i}{d_{\text{model}}}}$$

but computer could create problem so we did this 
$$e^{\frac{-2i}{d_{\text{model}}} \times \ln(10000)}$$

just following formula here above 

now torch.arange(0, d_model, 2): Steps by 2 (0, 2, 4, 6...till d_model-1)

 
have a look closely here 


pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

0::2: This means "start at column 0 and jump by 2". We fill all the even columns (0, 2, 4...) with Sine waves.

1::2: This means "start at column 1 and jump by 2". We fill all the odd columns (1, 3, 5...) with Cosine waves.


unsqueeze(0): Adds a batch dimension at the front. The shape goes from $(5000, 512)$ to $(1, 5000, 512)$. Now it perfectly matches.--- the shape of the output from our InputEmbeddings layer!register_buffer: This is a very important PyTorch command. It tells the model: "Hey, keep this pe matrix saved inside you, move it to the GPU if asked, but do not update it during training." It is pure, fixed math, not a learned weight.


now forward pass ->
just doing the stuff of adding position encoding to input embedding then apply droupout over over it 

note : my mistake i did ealie was forgetting he -ve sign  in the -math.log() because there we aposition * div_term) multiplying


$$\text{Final Vector} = \text{Embedding Vector} + \text{Positional Vector}$$


till now clear 


3.-- MultiHead Attention(self Attention)--

![alt text](image-8.png)
![alt text](image-9.png)

Self Attention and self attention with masking

assume man eat the animal here model nned to understand that man eated the animal not animal eated the man

so that is the need of self attention 


here we iwll look at 3 things query, key , value

query: what i am looking offer

key: what i can offered

value : what i actually offered

The Attention mechanism compares your Query against every Key. It calculates a "score" of how well they match.

in the above example eat you need to know what is refering too

node add image here 

the reSON IT CALLED SELF ATTENTION WE ARE ;OOKING IN THE SENTENCCE IN IT what it is reffering too thats why self attention


its relly simple for query ,key,value you mmake a table of seq_len*embedding(that is d_model)

ex image : ![alt text](image-1.png)

simple random genrate these value like  q=np.randn(L,d_k)
do for all three


![alt text](image.png)

first q is getting multiplied by transpose of using matmul thanks to it every row get mutiplied with with other coloum , i hope you know matrix multiplication :) 

second divide by root dk so that we can decrese varience becuse as we increase dimenion the varience increase that many times so did this thing
 apply softmax formula(exp(x)/sigma(exp()x) over it
 matmul with v 

self attention with masking

![alt text](image-2.png)
we just only add M for amsking

well for future refrenc ejust to tell you mask will look like this 

![alt text](image-3.png)

mask will look like this here replace 1 with 0 and 0 with -infinity

now why we will gived masked attention for decoder but what will happen with that is that now in outout we are not sending model the complete thing we are sending line by line word not the complete one softmax will convert the -inf to 0 .

some extra info that first thing we send in putput is <start> you can chatgpt about it

now we have attention but we need multihead atttention

multihead attention is simply getting prespective of lot of people it means now "it" word in above image judge the sentencce each word from different angles and give score on each angle

remember that heads given i paper = 8


we divide our 512 in 8 so each head will contian 64 but ine thing to  take care of that there are 3*512 of qkv   

so qkv -> (1,11,8,192) reminder for me  11 is seq length which here i assume

change shape to 1,8,11,64 for each no w q,k,v

![alt text](image-10.png)


at last :
![alt text](image-11.png)

	this is the code :
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self , d_model:int , h:int , dropout:float ) -> None :
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 'D_model is not visible' #assert this just looks at the condtion and throws error
        self.d_k = d_model // h
        self.w_q = Linear(d_model , d_model)
        self.w_k = Linear(d_model , d_model)
        self.w_v = Linear(d_model , d_model)
        self.w_o = Linear(d_model , d_model)
        self.dropout = nn.Dropout(dropout)

    # talking about static function in class , just one funcction for all instannce of class
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.dropout):
        # we donot try to put hardcode numbers
        d_k = query.shape[-1]
        #@ same as matmul like matrix multiplication
        attention_score = (query @ key.transpose(-2,-1)) /math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score=attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value) , attention_score

    def forward(self,q,k,v,mask):
        query = self.w_q(q) #(1,11,512)
        key = self.w_k(k)
        value = self.w_v(v)
        # i could have used reshape here instead of view but the reason for using view is that it changes in the object itself rather than creating a copy 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #(1,8,11,64)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2) 
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2) 
        x,self.attention_score = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        # (1,8,11,64)
        # x = x.transpose(1,2)    # (1,11,8,64)
        # combining the heads
        # x = x.view(x.shape[0],-1,self.h * self.d_k)
        # sometime i gorget this -1 so it simply is there to main the multiplication number
        #combining it with contingeour to ensure that all these allocation happen in one bloack of memory
        x = x.transpose(1,2).contiguous.().view(x.shape[0],-1,self.h * self.d_k)
        return self.w_o(x)
        
        
    some ai genrated explaination:
![alt text](image-12.png)

self.w_o is a Linear layer (a matrix multiplication) with the shape $(512, 512)$.When you multiply your stapled-together 512 vector by this $512 \times 512$ matrix, the math forces every single number to interact with every other number. It acts as the final "Judge" or "Synthesizer."It takes a little bit of insight from Head 1, mixes it with a little bit of insight from Head 4, and blends them into a final, unified 512-dimensional representation.





4.-----Layer Normalization----

I hope you know mean and variencce , mean simly the average and varience is simply the the avg distavnce b/w terms .

stand dard deviation is root( varience)


In normaization we just mean center our input by subtracting mean and divide by standard daviation for standarizing the data means the spread of the data. 


![alt text](image-13.png)

beta is there to shift the data

gamma is there to spread the data

why are we doing this what if the model not giving good output for the proper standard values so we are adding beta and gamma so that model can have some power of chaging these value and learn better 


code for this :

class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6 ) -> None :
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self , x ):
        mean = x.mean(dim=-1 , keepdim =true) #we want mean of last dimension 
        std = x.std(dim=-1,keepdim = True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias        
        
currenlty we are going taking one sentence what if in future we want to take 10 sentance so it wil req mean of batch it can itself handle batch


benifits of it:
faster trainign, stable training , nd covarience shift get reduced



---Feed Forward Network---

 This consist of two linear transforamtion with a RELU activation in b/w

 ![alt text](image-14.png)

 input=512 
 output=512
 inner layer = 2048

 this isthe code just a simple feed forward nothing much here to expalin 

 class FeedForwardBlock(nn.module):
    def __init__(self, d_model:int , d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self ,x):
        # b = self.linear_1(x)
        # s = torch.relu(b)
        # e = self.dropout(s)
        # z = self.linear_2(e)
        # Professional way same above thing
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
        

--Residual Connection -----

you can see residual connectio here form thsi image 
![alt text](image-15.png)


we will write it once and use it for all the transformer


you can have a look at the commects for exapliniation 


class ResidualConnection(nn.Module):
    def __init__(self , dropout:float ) ->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        # return x + self.norm(sublayer(x)) this is done what it simply mean is first send to sublayer it can be Multihead attention or it can be FeedForward layer normalize it then add
        # but for prevent over fitting we does it 
        # return x + self.dropout(self.norm(sublayer(x))) this is correct but acc to mit engneer for better
        return x + self.dropout(sublayer(self.norm(x)))

It is done to prevent vanishing gradient problem , have the earlier informaton if info get lost



7.----Encoder---

see the image around encoder you will see "N" is wrtten there

![alt text](image-16.png)

we will take the key and value from the encoder
and feed it to decoder 

First we will mke a single encode then for 6

here is only 1 block:

 class EncoderBlock(nn.Module):
     def __init__(self, self_attention: MultiHeadAttentionBlock , feed_forward_block:FeedForwardBlock , dropout:float )->None:
         super().__init__
         self.self_attention_block = self_attention_block #here i got the instance of multiheadattention class
         self.feed_forward_block = feed_forward_block
         self.residual_connections = nn.moduleList([ResidualConnection(dropout) for _ in range(2) ]) #making a instance of residual connection class but each layer of enccoder consit of two of these one for multihead other for feedforward
         # the nn.moduleList is used to store sub class in there

     def forward(self , x , src_mask):
         # the reason for the src msk is that ech sentence which is going in model in of different length so we add something in smll sentence to make them of corect length its <pad> chatgpt it  
         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask) ) # that k , q , v and mask is getting sended
         # 0 beacuse there are two residual connections
         x = self.residual_connection[1](x , self.feed_forward_block)
         return x     
          

noe stacking them one on anoehter

now attaching all of them just one mpre thing after passing it with 6 encoder we will normalize it 

class Encoder(nn.Module):
    # just assume that nnModuleList for now contain encoder of quantity 6
    def __init__(self, layers: nn.ModuleList )->None :
        super().__init__
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layers in self.layers:
            x = layer(x,mask)
        return self.norm(x)
        

---Decoder---

![alt text](image-17.png)

all same just ive key and value from encoder

it just repeat one thing to note is cross attention

some ai exxplanatio of these:
1. Masked Multi-Head Attention (Masked Self-Attention)What it does: Looks at tokens within the same sequence (typically the target output sequence) to understand how they relate to one another.The "Mask" part: It applies a mathematical mask to block the model from "looking ahead" at future tokens. It forces the model to predict a word based only on the words that came before it.Where it is used: Primarily in the Transformer Decoder during training to ensure the model learns to generate text naturally (autoregressively).Inputs: Queries (Q), Keys (K), and Values (V) all come from the same source (e.g., the decoder's own hidden states).2. Cross-AttentionWhat it does: Bridges two different sequences together. It uses the state of one sequence (usually the output) to "query" the information found in another sequence (the input).Where it is used: Only in Encoder-Decoder models (like standard machine translation architectures). It allows the decoder to look at the original source text while generating the translated output.Inputs:Queries (Q) come from the Decoder (the output being generated).Keys (K) and Values (V) come from the Encoder (the original input text)


class DecoderBlock(nn.module):
    def __init__(self,self_attention: MultiHeadAttentionBlock ,cross_attention: MultiHeadAttentionBlock , feed_forward_block:FeedForwardBlock , dropout:float )->None:
        super().__init__()
        self.self_attention_block = self_attention_block #here i got the instance of multiheadattention class
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3) ])

    def forward(self,x,encoder_output, src_mask , target_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(d, encoder_output,encoder_output ,src_mask))
        x = self.residual_connections[2](x , self.feed_forward_block)\
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList )->None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask, target_mask):
        for layers in self.layers:
            x = layer(x,encoder_output, src_mask , target_mask)
        return self.norm(x)


---Linear Layer---
![alt text](image-18.png)

its hedll simple a simple linear layer and a softmax one well actually the softmax layer is giving us the probablity of each number in voack it is telling the probB OF COMMING OF EACH NUMBER AND WE ARE CHOOSING THE HIHEST OF IT , thats why chggpt , gemini all these model s are also called the nexxt word predictor not real agi.


---9. Building the transformer---

just combining all ,simple :)



class Transformer(nn.Module):
    def __init__(self , encoder : Encoder , decoder : Decoder , src_embed: InputEmbeddings , target_embed:InputEmbeddings , src_pos: PositionalEncoding, target_embed:PositionalEncoding , projection_layer:ProjectionLayer)->None:
        super()__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encoder(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src , src_mask)

    def decoder(self , encoder_output, src_mask , target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(tgt)
        return self.decoder(tgt ,encoder_output , src_mask, target_mask)

    def project(self,x):
        return self.projection_layer(x)



----Building Transformer----


# Notice now we are making function
# we earlier disscussed difference in vacb and seq len
def build_transformer(src_vocab_size:int , target_vocab_size:int , src_seq_len:int , target_seq_len:int , d_model:int=512 , N:int=6 , h:int=8 , dropout:float=0.1,d_ff:int=2048) -> Transformer:
    src_embed = InputEmbedding(d_model,src_vocab_size)
    target_embed = InputEmbedding(_model , target_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model , target_seq_len)
    
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block=FeedForwardBlock(d_model, d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block ,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model , h , dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block=FeedForwardBlock(d_model, d_ff,dropout)
        decoder_block = EncoderBlock(encoder_self_attention_block ,decoder_cross_attention_block, feed_forward_block ,dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks)) #it got converted to module list we send it to encoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer=ProjectionLayer(d_model, target_vocab_size)
    transformer = Transformer(encoder , decoder , src_embed, target_embed ,src_pos, target_pos, projection_layer)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            # interesting info there is a "_" after uniform here it means change in place rather than creating a copy
            # the parameters who have dimention greater than one
            # the variance of the outputs of a layer needs to be exactly equal to the variance of its inputs. Simply put: If 512 numbers flow into a Linear layer, the 512 numbers that flow out shouldn't be drastically larger or drastically smaller.
    return transformer
    
        
        
That was the transformer now its time to test it lets do it




