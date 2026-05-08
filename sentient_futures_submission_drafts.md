# Sentient Futures Submission — 250-Word Response Drafts

**Prompt:** Describe what the project asked, what you did, and what you found
or built. We recommend writing for a smart non-specialist reader. [250 word
limit]

Three variations, all findings-led, written for a lay reader. Same closing
paragraph in each (Replacement B, dialed back from earlier overclaim about
fine-tuning recipes).

---

## Variation 1 — narrative, with metaphor

Imagine an AI taught to recognize animal calls. It can tell a robin from a
sparrow, but does it understand anything about animals, or has it just
memorized patterns? The Earth Species Project builds models like this to
help decode animal communication. We asked a simple question: when these
models are trained, what actually changes inside?

We studied four trained versions of their audio network plus an untrained
fifth as a baseline. Each network has thirteen processing layers stacked end
to end. We fed all five the same 600 sound clips and asked two questions of
every layer. First, how was it organizing its internal map of the sounds?
Second, how easily could a small readout classifier pull a category like
"bird" versus "mammal" out of it?

The pattern was striking. Coarse distinctions like animal versus music, or
bird versus mammal, snap into focus in the early-to-middle layers.
Distinctions between closely related species only appear in the deepest
layers. The closer two species sit on the evolutionary tree, the deeper you
have to go to tell them apart. The model has, in effect, rediscovered
evolution from sound alone.

Every trained version learned something the untrained baseline did not. The
differences were in how they organized that knowledge. One recipe sorted
animal class, taxonomic order, and species onto near-independent internal
axes, a clean hierarchy. The others picked up the same categories but
encoded them in overlapping, tangled ways. We built tools that make this
difference visible.

---

## Variation 2 — punchy, headline-led

What happens inside an AI when you train it on animal sounds? The Earth
Species Project develops audio models that identify species from recordings,
aiming to help decode animal communication. Their models are accurate, but
accuracy is a black box. We wanted to see whether the network has truly
learned something about animals, or just memorized which patterns go with
which label.

We took four trained versions of their audio network plus a fifth,
untrained, as a control. Each network passes sound through thirteen
processing layers in sequence. We ran 600 clips through every layer of every
model, then asked two things of each layer. Is it organizing animal sounds
in a meaningful shape? Can a small readout classifier pull labels like
"bird" or "mammal" out of it?

The answers came back in a beautiful pattern. Coarse categories like animal
versus music or bird versus mammal sort out in the first half of the
network. Fine distinctions between similar species, like the great tit and
the Turkestan tit, only resolve in the deepest layers. The closer two
species sit on the evolutionary tree, the deeper the network has to go to
separate them. Network depth tracks evolutionary time.

Every trained version learned something the untrained baseline did not. The
differences were in how they organized that knowledge. One recipe sorted
animal class, taxonomic order, and species onto near-independent internal
axes, a clean hierarchy. The others picked up the same categories but
encoded them in overlapping, tangled ways. We built tools that make this
difference visible.

---

## Variation 3 — concrete examples up front

Recognizing birdsong is now an AI's job. Groups like the Earth Species
Project build models that can listen to a recording and name the species.
The long-term goal is more ambitious: using these models to help decode
animal communication. But a high score on a test does not tell you whether
the AI truly understands what it hears. We set out to look inside.

We studied four trained versions of their audio network plus an untrained
fifth as a baseline. Each network feeds a sound through thirteen processing
layers in sequence, and we examined what each one was doing. We did this two
ways: measuring the geometric shape of the network's internal map of the
sounds, and training tiny classifiers to read categories like "bird" or
"sparrow" out of each layer.

A striking pattern appeared. Coarse categories, animal versus non-animal or
bird versus mammal, sort themselves out early. Distinctions between closely
related species, like the house sparrow and the tree sparrow, only appear in
the deepest layers. The closer two species sit on the evolutionary tree, the
more processing depth the model needs to tell them apart. Evolutionary
distance is encoded in network depth.

Every trained version learned something the untrained baseline did not. The
differences were in how they organized that knowledge. One recipe sorted
animal class, taxonomic order, and species onto near-independent internal
axes, a clean hierarchy. The others picked up the same categories but
encoded them in overlapping, tangled ways. We built tools that make this
difference visible.
