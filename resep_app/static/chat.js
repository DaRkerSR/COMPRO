document.addEventListener('DOMContentLoaded', function(){
  const toggle = document.querySelector('.sc-toggle');
  const windowEl = document.querySelector('.sc-window');
  const closeBtn = document.querySelector('.sc-close');
  const form = document.querySelector('.sc-form');
  const messages = document.querySelector('.sc-messages');
  const input = document.querySelector('.sc-input');

  function appendMessage(text, who){
    const el = document.createElement('div');
    el.className = 'sc-message ' + who;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  toggle.addEventListener('click', ()=>{
    windowEl.classList.toggle('hidden');
  });
  if(closeBtn) closeBtn.addEventListener('click', ()=> windowEl.classList.add('hidden'));

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const text = input.value.trim();
    if(!text) return;
    appendMessage(text, 'user');
    input.value = '';
    try{
      const res = await fetch('/chat', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({message: text})
      });
      const data = await res.json();
      appendMessage(data.reply || 'Maaf, tidak ada balasan.', 'bot');
    }catch(err){
      appendMessage('Terjadi kesalahan saat menghubungi server.', 'bot');
    }
  });

});
