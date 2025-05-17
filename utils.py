import pretty_midi
import numpy as np

def piano_roll_to_midi(piano_roll, fs=100):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    piano_roll = (piano_roll > 0.5).astype(np.uint8)

    for pitch in range(128):
        notes = np.where(piano_roll[:, pitch])[0]
        if len(notes) == 0:
            continue
        start = notes[0]
        for i in range(1, len(notes)):
            if notes[i] != notes[i-1] + 1:
                end = notes[i-1]
                instrument.notes.append(pretty_midi.Note(
                    velocity=100, pitch=pitch,
                    start=start/fs, end=end/fs))
                start = notes[i]
        end = notes[-1]
        instrument.notes.append(pretty_midi.Note(
            velocity=100, pitch=pitch,
            start=start/fs, end=end/fs))
    midi.instruments.append(instrument)
    return midi
