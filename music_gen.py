#!/usr/bin/env python
# coding: utf-8

# In[1]:


from music21 import converter,instrument,note,chord,stream
import glob
import pickle
import numpy as np


# ### Read a MIDI file

# In[2]:


midi = converter.parse('midi_songs/EyesOnMePiano.mid')


# In[3]:


midi.show('midi')


# In[ ]:


midi.show('text')


# In[20]:


all_els = midi.flat.notes


# In[21]:


len(all_els)


# In[27]:


notes_demo= []
for e in all_els:
        # If the element is a Note, store its pitch
        if isinstance(e,note.Note):
            notes_demo.append(str(e.pitch))
        
        # if the element is a chord, split each note of chord and join them with '+'
        elif isinstance(e,chord.Chord):
            notes_demo.append('+'.join(str(n) for n in e.normalOrder))


# ## Preprocessing the MIDI files

# In[33]:


notes = []
for file in glob.glob('midi_songs/*.mid'):
    midi = converter.parse(file)  # Conver a file into stream.Score object
    
    print('parsing %s'%(file))
    
    all_els = midi.flat.notes
    for e in all_els:
        # If the element is a Note, store its pitch
        if isinstance(e,note.Note):
            notes.append(str(e.pitch))
        
        # if the element is a chord, split each note of chord and join them with '+'
        elif isinstance(e,chord.Chord):
            notes.append('+'.join(str(n) for n in e.normalOrder))


# In[34]:


with open('notes','wb') as f:
    pickle.dump(notes,f)


# In[38]:


with open('notes','rb') as f:
    pickle.load(f)


# In[86]:


n_vocab=len(set(notes))
print(n_vocab)


# In[42]:


print(notes[200:290])


# ## Preparing Sequential data

# In[52]:


seq_len = 100


# In[46]:


pitch_names = sorted(set(notes))


# In[47]:


ele_to_int =dict((ele,num) for num,ele in enumerate(pitch_names))


# In[88]:


network_input = []
network_output = []


# In[89]:


for i  in range(len(notes) - seq_len):
    seq_in = notes[i:i+seq_len]
    seq_out = notes[i+seq_len]
    
    network_input.append([ele_to_int[ch] for ch in seq_in])
    network_output.append(ele_to_int[seq_out])


# In[90]:


n_patterns = len(network_input)


# In[91]:


network_input = np.reshape(network_input,(n_patterns,seq_len,1))


# In[92]:


norm_network_input = network_input/float(n_vocab)


# In[93]:


from keras.utils import np_utils
network_output=np_utils.to_categorical(network_output)


# In[94]:


print(norm_network_input.shape)
print(network_output.shape)


# # *Building the MODEL*

# In[95]:


from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[96]:


model = Sequential()
model.add(LSTM(
    units =512,
    input_shape = (norm_network_input.shape[1],1),
    return_sequences = True
))
model.add(Dropout(rate=0.3))
model.add(LSTM(
        units = 512,
        return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab,activation='softmax'))


# In[97]:


model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()


# 
# ## Training the Model

# In[100]:


check_p = ModelCheckpoint('model.hdf5',monitor='loss',verbose=0,save_best_only=True,mode='min')
model.load_weights('new_weights.hdf5')


# ##  **Predictions**

# In[129]:


network_input = []
for i  in range(len(notes) - seq_len):
    seq_in = notes[i:i+seq_len]
    network_input.append([ele_to_int[ch] for ch in seq_in])


# In[130]:


start = np.random.randint(len(network_input)-1)
#print(network_input[start])


# In[131]:


# int_to_el MAPPING
int_to_ele=dict((num,el) for num,el in enumerate(pitch_names))

# Initial Pattern

pattern = network_input[start]
prediction_output = []

# Generate 200 elements

for note_index in range(200):
    prediction_input = np.reshape(pattern,(1,len(pattern),1))
    prediction_input = prediction_input/float(n_vocab)
    
    prediction = model.predict(prediction_input,verbose=0)
    
    idx = np.argmax(prediction)
    result = int_to_ele[idx]
    prediction_output.append(result)
    
    pattern.append(idx)
    pattern =pattern[1:]


# In[132]:


offset = 0 #Time
output_notes = []
for patt in prediction_output:
    
    # if Chord:-----
    if ('+' in patt or patt.isdigit()):
        notes_in_chord = patt.split('+')
        temp_notes = []
        for curr in notes_in_chord:
            new_note = note.Note(int(curr))  # Create Note Obj for each note in the chord
            new_note.storedInstrument = instrument.Piano()
            temp_notes.append(new_note)
            
        new_chord = chord.Chord(temp_notes) # Create the Chord()  obj from the list of notes
        new_chord.offset = offset
        output_notes.append(new_chord)
    
    # if Note:-----
    else:
        new_note = note.Note(patt)
        new_note.offset =offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    
    offset+=0.5
    


# In[133]:


# Create a stream object from the generated notes
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi',fp='test_output.mid')


# In[134]:


midi_stream.show('midi')

