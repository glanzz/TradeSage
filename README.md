# TradeSage
TradeSage is a GenAI-powered financial advisory assistant that provides intelligent, context-aware insights for your next trade. It combines live market sentiment, semantic memory, and deep domain understanding from The Alchemy of Finance to simulate a smart trading partner.

### Architecture

<img src="/architecture.png">

### UI
<img width="40%" src="/ui.png">


### Components
#### Data pre-processing layer
- Preprocessing layer using the PDF to text with `docling` & QA generation using `valhalla/t5-base-e2e-qg` for directed task

#### Training
- Transformer architecture build from scratch:
<pre>
Model: "transformer"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 inputs (InputLayer)         [(None, None)]               0         []                            
                                                                                                  
 dec_inputs (InputLayer)     [(None, None)]               0         []                            
                                                                                                  
 enc_padding_mask (Lambda)   (None, 1, 1, None)           0         ['inputs[0][0]']              
                                                                                                  
 encoder (Functional)        (None, None, 256)            1362944   ['inputs[0][0]',              
                                                                     'enc_padding_mask[0][0]']    
                                                                                                  
 look_ahead_mask (Lambda)    (None, 1, None, None)        0         ['dec_inputs[0][0]']          
                                                                                                  
 dec_padding_mask (Lambda)   (None, 1, 1, None)           0         ['inputs[0][0]']              
                                                                                                  
 decoder (Functional)        (None, None, 256)            1890304   ['dec_inputs[0][0]',          
                                                                     'encoder[0][0]',             
                                                                     'look_ahead_mask[0][0]',     
                                                                     'dec_padding_mask[0][0]']    
                                                                                                  
 outputs (Dense)             (None, None, 1206)           309942    ['decoder[0][0]']             
                                                                                                  
==================================================================================================
Total params: 3563190 (13.59 MB)
Trainable params: 3563190 (13.59 MB)
Non-trainable params: 0 (0.00 Byte)
</pre>

#### Response Post-Processing
- Semantic memory with milvus for storage & sentence transformer model: `all-MiniLM-L6-v2` for embedding question historical questions
- Context manager to pass context along with the user question using relevant past answers in conversation history maintaining the context size with conversational logs
- Inital prompts generation to start with for users from financial feeds
- Text refinement to fix grammatical errors using `t5-flan-small`
- UI for interaction with bot using `streamlit`


### Features
- LLM-Powered Trade Advice: Generates nuanced trade insights using transformer-based models.

- Live Financial News as Context: Automatically injects a news-driven initial prompt to ground the conversation in real-time events.

- Semantic Memory with Milvus: Maintains chat history using vector embeddings for contextual continuity.

- Domain Knowledge: Trained on curated QA pairs extracted from The Alchemy of Finance to reflect strategic depth and reflexive thinking.

- Contextual QA Generation: Uses valhalla/t5-base-e2e-qg with stride-based segmentation (stride = 1) for high-quality QA generation.

- Text Refinement via flan-t5-small: Enhances clarity and tone of responses, ensuring professional and readable output.

- Interactive UI with Streamlit: Offers a chat-like experience for real-time interaction.



### Sample Conversation
<pre>
You:  Current Gold price $2,400 an ounce, would you recommend adding gold to my current portfolio at this price?
Bot: this would effectively meant a high exchange rate policy .

You:  With silver trading at about $30 per ounce, how does it compare to gold as an inflation hedge?
Bot: this would involve not only putting our own house in order but also bringing a new international financial order into existence which is appropriate to the altered circumstances .


You:  The GBPUSD trading and 1.3194. That's within six pips of the high for the day at 1.3200. The high price for the year is also being approached. That level came in at 1.32067 reached on April 3.  Move above that level and a swing area between 1.3221 to 1.3245 would be targeted.
Bot: if so , we may yet see the bull market of a lifetime .

You:  Bank stocks have P/E ratio around 12-15. Are they undervalued under current interest rate environment ? Current intrest rate 7.128%.
Bot: with reduced exposure , i can recover and regroup more easily . i accept the opec agreement as a major event.

You:  Bitcoin has been fluctuating between $65,000-$70,000 lately - is this a good entry point for someone new to crypto?
Bot: at present , the odds are that the market will pass the test , but it would be best not to be exposed while the issus is decided .

You:  The VIX volatility index is around 20 - what does that tell you about market sentiment also the small caps funds have outperformed large cap funds lately.
Bot: as a consequence , we expect a period of continuing turmoil in financial markets , although the focus of attention may shift from the stock market to the currency market and the bond market and , eventually , to the market for precious metals . the ideal of an international currency and an international central bank has few supporters .



You:  Is it a sign of recession ?
Bot: it is difficult to see how such a point could be reached as long as the federal government continues to borrow at the pres ent rate . a negative resolution would involve increasing financial instability , protectionism , a worldwide recessioa , and a flight from financial to liquid assets .

You:  exit
Bot: Exiting chatbot. Goodbye!

</pre>

### Screenshots
<img src="/responses.png">


### Installation

#### Clone the repo
```bash
git clone https://github.com/glanzz/TradeSage.git
cd TradeSage
```

#### Install dependencies
```bash
pip install -r requirements.lock
```

#### Ensure Milvus is running
#### See Milvus docs for setup: https://milvus.io/docs

#### Run the app
```bash
streamlit run app/main.py
```


### License
MIT License. See LICENSE for details.
