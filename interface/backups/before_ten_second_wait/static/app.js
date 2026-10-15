'use strict';
const $ = id => document.getElementById(id);
let active = null, timer = null, imageName = 'screenshot.png';
const captions = {
  'screenshot.png': 'Captured viewport · 1366 × 768. Confirm that this is the intended webpage.',
  'model_input.png': 'The actual image supplied to the selected CNN. The exact float tensor is saved with the run.',
  'phishing_overlay.png': 'Phishing-class Grad-CAM · warmer areas have greater positive activation for this target.',
  'legitimate_overlay.png': 'Legitimate-class Grad-CAM · normalised independently from the phishing map.'
};
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request failed.');
  return data;
}
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
function showImage(name) {
  imageName=name;
  if (!active?.result) return;
  const url=`/files/${active.id}/${name}`;
  $('evidence-image').src=url; $('evidence-image').alt=captions[name];
  $('download-image').href=url; $('caption').textContent=captions[name];
  $('evidence-image').parentElement.classList.toggle('model-input',name!=='screenshot.png');
  document.querySelectorAll('[data-image]').forEach(button=>{
    const selected=button.dataset.image===name;
    button.classList.toggle('selected', selected); button.setAttribute('aria-selected',String(selected));
  });
}
function render(job) {
  active=job;
  const busy=['queued','running'].includes(job.status);
  $('run').disabled=busy;
  message(job.status==='failed' ? job.error : job.stage, job.status==='failed');
  if(job.status!=='complete') { $('output').hidden=true; $('empty').hidden=false; return; }
  $('empty').hidden=true; $('output').hidden=false;
  const r=job.result;
  $('page-title').textContent=r.capture.title || 'Website analysis'; $('result-url').textContent=r.url;
  $('verdict').textContent=r.prediction; $('verdict').classList.toggle('phishing',r.prediction==='PHISHING');
  $('score').textContent=r.phishing_score.toFixed(4); $('score-bar').style.width=`${r.phishing_score*100}%`;
  $('threshold').textContent=`Phishing threshold ≥ ${r.threshold.toFixed(3)}`;
  $('model').textContent=`Model ${r.model_number}`;
  $('input-size').textContent=`EfficientNetV2B0 · ${r.checkpoint.width} × ${r.checkpoint.height}`;
  $('page-type').textContent=r.router.page_type==='LOGIN'?'Login page':'General webpage';
  $('warnings').hidden=!r.capture.warnings.length;
  $('warnings').textContent=r.capture.warnings.length ? 'Capture needs review: '+r.capture.warnings.join(' ')+' The classification describes the captured image.' : '';
  $('download-csv').href=`/files/${job.id}/prediction.csv`; $('download-json').href=`/files/${job.id}/result.json`;
  const rows=[['Run ID',r.id],['Created (UTC)',r.created_at],['Final captured URL',r.capture.final_url],['HTTP status',r.capture.http_status ?? 'Unavailable'],['Router reasons',JSON.stringify(r.router.reasons)],['Checkpoint',r.checkpoint.checkpoint],['SHA-256',r.checkpoint.sha256],['Threshold',r.threshold],['Preprocessing',r.checkpoint.preprocessing],['Capture mode','Headless Chromium; 1366 × 768 viewport; 3 s render wait'],['Grad-CAM layer',r.explanation.PHISHING.layer],['Phishing map empty',r.explanation.PHISHING.empty_heatmap],['Legitimate map empty',r.explanation.LEGITIMATE.empty_heatmap],['Forward parity error',Math.max(...Object.values(r.explanation).map(x=>x.forward_parity_error))],['Score parity error',r.component_score_difference]];
  $('details').replaceChildren();
  for(const [key,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=String(value);$('details').append(dt,dd);}
  showImage(imageName);
}
async function refreshHistory(){
  const jobs=await api('/api/jobs'); $('history').replaceChildren();
  if(!jobs.length){const p=document.createElement('p');p.textContent='No analyses yet.';$('history').append(p);return;}
  for(const job of jobs.slice(0,25)){
    const button=document.createElement('button');let host=job.url;try{host=new URL(job.url).hostname;}catch{}
    button.textContent=host;button.title=job.url;const small=document.createElement('small');small.textContent=job.result?`Model ${job.result.model_number} · ${job.result.prediction}`:job.status;button.append(small);
    button.addEventListener('click',()=>watch(job.id));$('history').append(button);
  }
}
async function watch(id){
  clearTimeout(timer);
  try { const job=await api(`/api/jobs/${id}`);render(job);await refreshHistory();
    if(['queued','running'].includes(job.status))timer=setTimeout(()=>watch(id),1200);
  } catch(error){message(error.message,true);$('run').disabled=false;}
}
$('analysis-form').addEventListener('submit',async event=>{
  event.preventDefault();$('run').disabled=true;message('Starting analysis…');
  try {const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('url').value})});imageName='screenshot.png';await watch(job.id);}
  catch(error){message(error.message,true);$('run').disabled=false;}
});
document.querySelectorAll('[data-image]').forEach(button=>button.addEventListener('click',()=>showImage(button.dataset.image)));
refreshHistory().catch(error=>message(error.message,true));
