/* Local UI: no external fonts, icon libraries or scripts required. */
document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);
  const toast = message => { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(window.toastTimer); window.toastTimer = setTimeout(() => $('toast').hidden = true, 5000); };
  $('theme-toggle').addEventListener('click', () => { const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = theme; try { localStorage.setItem('theme', theme); } catch(e) {} });
  const closeSidebar = () => { document.body.classList.remove('sidebar-open'); $('backdrop').hidden = true; $('menu-toggle').setAttribute('aria-expanded','false'); };
  $('menu-toggle').addEventListener('click', () => { const open = document.body.classList.toggle('sidebar-open'); $('backdrop').hidden = !open; $('menu-toggle').setAttribute('aria-expanded',String(open)); });
  $('backdrop').addEventListener('click', closeSidebar);
  document.addEventListener('keydown', e => { if(e.key === 'Escape') closeSidebar(); });
  $('help-toggle').addEventListener('click', () => $('help-dialog').showModal());
  document.querySelectorAll('[data-close-dialog]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', e => { if(e.target === dialog) { const r = dialog.getBoundingClientRect(); if(e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)dialog.close(); } }));
  const form = $('question-form'), input = $('question-input');
  const examples = {
    employee: ['新员工入职需要准备哪些材料？', '公司的考勤制度是怎样的？', '试用期多久？'],
    product: ['公司的主要产品有哪些？', '激光打标设备有哪些特点？', '介绍红外辐射的应用场景。'],
    professional: ['什么是半导体光刻工艺？', '红外探测器的工作原理是什么？', '第三代半导体材料有哪些优势？']
  };
  document.querySelectorAll('.kb-card:not(:disabled)').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('.kb-card').forEach(card => { card.classList.toggle('is-selected',card === button); card.setAttribute('aria-pressed',String(card === button)); });
    $('category-input').value = button.dataset.category;
    $('selected-label').textContent = button.dataset.label;
    input.placeholder = button.dataset.category === 'professional' ? '请输入专业知识问题' : '请输入你的问题';
    document.querySelectorAll('[data-question]').forEach((b,i) => { b.dataset.question = examples[button.dataset.category][i]; b.firstChild.textContent = examples[button.dataset.category][i] + ' '; });
  }));
  document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => { input.value = button.dataset.question; input.focus(); }));
  let submitting = false;
  async function submitQuestion(question) {
    if(submitting || !question.trim()) return;
    submitting = true; const submit = form.querySelector('[type=submit]'); submit.disabled = true;
    const original = submit.innerHTML; submit.textContent = '正在提交…';
    try {
      const body = new FormData(form); body.set('query',question.trim());
      const response = await fetch('/solve',{ method:'POST', body }); const data = await response.json();
      if(!response.ok || !data.run_id) throw new Error(data.message || '提交失败，请重试');
      location.href = `/session/${encodeURIComponent(data.run_id)}?${new URLSearchParams({prompt:question.trim(),category:body.get('category')})}`;
    } catch(e) { toast('暂时无法提交问题，请检查连接后重试。'); submitting = false; submit.disabled = false; submit.innerHTML = original; }
  }
  form.addEventListener('submit', e => { e.preventDefault(); submitQuestion(input.value); });
  input.addEventListener('keydown', e => { if(e.key==='Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });

  // Escape raw model output before applying a small Markdown presentation subset.
  const escape = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const inline = text => escape(text).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  function markdown(text) {
    const lines = String(text).replace(/\r/g,'').split('\n'); let html = '', list = null;
    const closeList = () => { if(list)html += `</${list}>`; list = null; };
    const cells = line => line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(c => c.trim());
    for(let i=0;i<lines.length;i++) {
      const line = lines[i];
      if(/^```/.test(line)) { closeList(); const code=[]; while(++i<lines.length && !/^```/.test(lines[i]))code.push(lines[i]); html += '<pre><code>'+escape(code.join('\n'))+'</code></pre>'; continue; }
      if(line.includes('|') && i+1<lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i+1]) && cells(lines[i+1]).every(c => /^:?-{3,}:?$/.test(c))) {
        closeList(); html += '<div class="table-wrap"><table><thead><tr>'+cells(line).map(c => '<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>'; i++;
        while(i+1<lines.length && lines[i+1].includes('|') && lines[i+1].trim()) { i++; html += '<tr>'+cells(lines[i]).map(c => '<td>'+inline(c)+'</td>').join('')+'</tr>'; }
        html += '</tbody></table></div>'; continue;
      }
      const item = line.match(/^\s*(?:([-*])\s+|\d+[.)、]\s+)(.+)$/);
      if(item) { const tag = item[1] ? 'ul' : 'ol'; if(list!==tag) { closeList(); list=tag; html += `<${tag}>`; } html += '<li>'+inline(item[2])+'</li>'; continue; }
      closeList(); if(!line.trim())continue;
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if(heading) { const level = Math.min(heading[1].length+1,4); html += `<h${level}>${inline(heading[2])}</h${level}>`; }
      else if(/^>\s?/.test(line)) html += '<blockquote>'+inline(line.replace(/^>\s?/,''))+'</blockquote>';
      else if(/^[-*_]{3,}\s*$/.test(line)) html += '<hr>';
      else html += '<p>'+inline(line)+'</p>';
    }
    closeList(); return html;
  }
  const session = $('chat-session'); if(!session)return;
  let cursor=0, done=false, found=false, retries=0;
  function renderResult(data) {
    found=true; $('answer-content').innerHTML = markdown(data.answer || '没有返回回答。');
    const refs = Array.isArray(data.references) ? data.references : [];
    const category = $('category-input')?.value;
    $('answer-status').textContent = refs.length ? '回答已生成 · 可查看下方引用来源' : (category === 'professional' ? '大模型回答已生成 · 无知识库引用' : '回答已生成');
    const references = $('references'); references.replaceChildren(); references.hidden = !refs.length;
    if(refs.length) {
      const title=document.createElement('h3'); title.textContent=`参考来源 · ${refs.length} 个片段`; references.append(title);
      const list=document.createElement('div'); list.className='reference-list'; references.append(list);
      refs.forEach(ref => { const detail=document.createElement('details'); detail.className='reference-item'; const summary=document.createElement('summary'); summary.textContent=`[${ref.ref_id}] ${ref.doc_name || ref.file_path || '知识库资料'}`; const preview=document.createElement('p'); preview.textContent=ref.preview_text || '暂无片段预览'; detail.append(summary,preview); list.append(detail); });
    }
    const paths = [...new Set(refs.flatMap(r => r.image_paths || []))].filter(p => typeof p==='string' && /^\/(?:static\/img|img)\//.test(p));
    $('answer-images').replaceChildren(); $('answer-images').hidden=!paths.length;
    if(paths.length) {
      const heading=document.createElement('h3'); heading.textContent='相关资料图片'; const grid=document.createElement('div'); grid.className='image-grid'; $('answer-images').append(heading,grid);
      paths.forEach((path,index) => { const button=document.createElement('button'); button.type='button'; button.setAttribute('aria-label',`放大资料图片 ${index+1}`); const img=document.createElement('img'); img.src=path.startsWith('/img/') ? '/static'+path : path; img.alt=`资料图片 ${index+1}`; img.loading='lazy'; img.addEventListener('error',()=>{button.remove();if(!grid.children.length)$('answer-images').hidden=true;}); button.append(img); grid.append(button); button.addEventListener('click',()=>{$('preview-image').src=img.src;$('image-dialog').showModal();}); });
    }
    $('related-list').replaceChildren(); const questions=Array.isArray(data.related_questions)?data.related_questions:[]; $('related-questions').hidden=!questions.length;
    questions.forEach(q => { const button=document.createElement('button'); button.type='button'; button.textContent=q+' ↗'; button.addEventListener('click',()=>submitQuestion(q)); $('related-list').append(button); });
  }
  async function poll() {
    if(done)return;
    try {
      const response = await fetch(`/logs/${encodeURIComponent(session.dataset.runId)}?cursor=${cursor}`);
      const data=await response.json(); if(!response.ok || data.status!=='ok')throw new Error('会话不可用'); retries=0;
      for(const item of data.logs || []) {
        const text=item.pretty || ''; const separator=text.indexOf('|'); const content=separator>=0?text.slice(separator+1).trim():text;
        if(text.includes('phase=result')) { try { renderResult(JSON.parse(content)); } catch(e) { toast('回答格式异常，请查看检索记录。'); } }
        else if(text.includes('phase=answer') && !found) { $('answer-content').innerHTML=markdown(content); found=true; }
        else if(text.includes('phase=error')) { $('answer-content').textContent=content; $('answer-status').textContent='本次问答未完成'; found=true; }
        else { const p=document.createElement('p'); p.textContent=content; $('retrieval-log').append(p); }
      }
      cursor=data.next_cursor ?? cursor; done=Boolean(data.done);
      if(done && !found) { $('answer-content').textContent='这条历史对话没有可读取的回答，可以在下方重新提问。'; $('answer-status').textContent='暂无回答'; }
    } catch(e) { retries++; if(retries>=3) { done=true; $('answer-status').textContent='暂时无法读取会话'; if(!found)$('answer-content').textContent='会话记录不存在或服务连接中断。请刷新页面，或返回首页重新提问。'; } }
    if(!done)setTimeout(poll,1200);
  }
  if(session.dataset.invalid==='true') { $('answer-status').textContent='历史记录不可用'; $('answer-content').textContent='这条历史对话的回答文件未随项目提供。你可以在下方重新发送问题。'; }
  else poll();
});
