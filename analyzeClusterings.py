from music21 import *
us = environment.UserSettings()
us.create()
import numpy as np
environment.set("musescoreDirectPNGPath", "C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe")
environment.set('musicxmlPath', "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe")
environment.set('lilypondPath', "C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe")

def reassemble_patterns(labelling, test_occs, pOccs):
    found_patterns = []
    occ_name_to_pattern = {}
    for pat_num in list(set(labelling)):
        # if pat_num == -1:
        #     continue
        idxs = [i for i, x in enumerate(labelling) if x == pat_num]
        occ_names = [test_occs[i] for i in idxs]
        occs = [pOccs[n] for n in occ_names]
        found_patterns.append(occ_names)
        for on in occ_names:
            occ_name_to_pattern[on] = pat_num
    return found_patterns, occ_name_to_pattern

def list_notes(pat, pOccs):
    for occ_name in pat:
        occ = pOccs[occ_name]
        noteDurs = [round(float(x.quarterLength), 5) for x in occ.score.notes.stream()]
        noteNums = [x.pitch.midi for x in occ.score.notes.stream()]
        print(str(list(zip(noteNums, noteDurs))), occ.type)

emb_pats, emb_guide = reassemble_patterns(emb_labellings[3], test_occs, pOccs)
real_pats, real_guide = reassemble_patterns(labels_true, test_occs, pOccs)

EXAMPLE_PAT = real_pats[2]
EXAMPLE_EMB = emb_pats[3]
EXAMPLE_REAL = real_pats[17]

EXAMPLE_ARTIFICIAL = emb_pats[18]

for pat in emb_pats:
    print([real_guide[x] for x in pat])
for pat in real_pats:
    print([emb_guide[x] for x in pat])
list_notes(emb_pats[1], pOccs)

for i, occ_name in enumerate(EXAMPLE_ARTIFICIAL):
    score = pOccs[occ_name].score.notes
    for note in score:
        note.lyric = ''
    fname = 'pics/artifpat_{}'.format(i)
    conv = converter.subConverters.ConverterLilypond()
    conv.write(score.stream(), fmt='lilypond', fp=fname, subformats=['png'], dpi=500)

with open('paperfigs.pik', 'wb') as f:
    pickle.dump((emb_labellings, labels_true, test_occs, "labellings 4 and 3 were the ones used for figures"), f, -1)
