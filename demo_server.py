import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer


import re
import numpy as np
from util import audio

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="160" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(req.params.get('text'),mel_targets=mel_targets, reference_mel=reference_mel)
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/demo', UIResource())


if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=8080)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference_audio', default=None, help='Reference audio path')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  print(hparams_debug_string())

  is_teacher_force = False
  mel_targets = args.mel_targets
  reference_mel = None
  if args.mel_targets is not None:
    is_teacher_force = True
    mel_targets = np.load(args.mel_targets)
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.checkpoint, args.reference_audio)
  base_path = get_output_base_path(args.checkpoint)

  if args.reference_audio is not None:
    ref_wav = audio.load_wav(args.reference_audio)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    path = '%s_ref-%s.wav' % (base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
    alignment_path = '%s_ref-%s-align.png' % (base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
  else:
    if hparams.use_gst:
      print("*******************************")
      print("TODO: add style weights when there is no reference audio. Now we use random weights, " + 
             "which may generate unintelligible audio sometimes.")
      print("*******************************")
      path = '%s_ref-randomWeight.wav' % (base_path)
      alignment_path = '%s_ref-%s-align.png' % (base_path, 'randomWeight')
    else:
      raise ValueError("You must set the reference audio if you don't want to use GSTs.")

#   with open(path, 'wb') as f:
#     print('Synthesizing: %s' % args.text)
#     print('Output wav file: %s' % path)
#     print('Output alignments: %s' % alignment_path)
#     f.write(synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path))

  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
  synthesizer.load(os.environ['CHECKPOINT'])
