/* Progressive enhancement: content and SVGs remain readable without this file. */
(() => {
 'use strict';
 const $=(s,root=document)=>root.querySelector(s), $$=(s,root=document)=>[...root.querySelectorAll(s)];
 const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
 const menu=$('.menu-toggle'),sidebar=$('.sidebar'),shade=$('.mobile-shade');
 function closeMenu(){sidebar?.classList.remove('open');shade?.classList.remove('visible');menu?.setAttribute('aria-expanded','false');document.body.style.overflow='';}
 menu?.addEventListener('click',()=>{const opened=sidebar.classList.toggle('open');shade.classList.toggle('visible',opened);menu.setAttribute('aria-expanded',String(opened));document.body.style.overflow=opened?'hidden':'';});
 shade?.addEventListener('click',closeMenu);
 document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMenu();});
 const selected=$('.nav-link.active');if(selected&&sidebar)sidebar.scrollTop=Math.max(0,selected.offsetTop-sidebar.clientHeight*.48);
 $$('.copy-code').forEach(button=>button.addEventListener('click',async()=>{
  const text=$('code',button.closest('.codebox')).textContent;
  let success=false;
  try{await navigator.clipboard.writeText(text);success=true;}catch{
   const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.left='-10000px';document.body.append(area);area.select();success=document.execCommand('copy');area.remove();
  }
  button.textContent=success?'복사됨':'복사 실패';setTimeout(()=>button.textContent='복사',1800);
 }));
 let ticking=false;
 const headings=$$('.article h2, .article h3');
 function reading(){
  const max=document.documentElement.scrollHeight-window.innerHeight;
  const bar=$('.progress');if(bar)bar.style.width=(max?Math.min(100,window.scrollY/max*100):0)+'%';
  let current=headings[0];for(const h of headings){if(h.getBoundingClientRect().top<155)current=h;else break;}
  $$('.right-toc a').forEach(a=>a.classList.toggle('active',current&&decodeURIComponent(a.hash.slice(1))===current.id));ticking=false;
 }
 window.addEventListener('scroll',()=>{if(!ticking){requestAnimationFrame(reading);ticking=true;}},{passive:true});reading();
 const dialog=$('.search-dialog'), input=$('input',dialog||document), results=$('.search-results'),status=$('.search-status');
 let index=null,indexPromise=null;
 async function loadIndex(){if(index)return index;if(!indexPromise)indexPromise=fetch((document.body.dataset.root||'./')+'search-index.json?v='+(document.body.dataset.searchVersion||'current')).then(r=>{if(!r.ok)throw Error('search');return r.json();}).then(data=>index=data).catch(error=>{indexPromise=null;throw error;});return indexPromise;}
 async function openSearch(){closeMenu();dialog.showModal();input.focus();try{await loadIndex();search();}catch{status.textContent='검색 색인을 불러오지 못했습니다. 목차나 브라우저 찾기를 사용해 주세요.';}}
 $$('.search-open').forEach(b=>b.addEventListener('click',openSearch));
 $('.close-search')?.addEventListener('click',()=>dialog.close());
 dialog?.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
 document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();dialog.open?dialog.close():openSearch();}});
 function search(){
  if(!index)return;const q=input.value.trim().toLocaleLowerCase();results.replaceChildren();
  if(!q){status.textContent='전체 '+index.length+'개 문서에서 검색합니다. 용어나 함수 이름을 입력하세요.';return;}
  const terms=q.split(/\s+/).filter(Boolean);
  const matches=index.map(d=>{const t=d.text.toLocaleLowerCase(),title=d.title.toLocaleLowerCase();return {d,score:terms.every(w=>t.includes(w)||title.includes(w))?terms.reduce((v,w)=>v+(title.includes(w)?20:1)+(t.match(new RegExp(w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'g'))||[]).length,0):0};}).filter(x=>x.score).sort((a,b)=>b.score-a.score).slice(0,20);
  status.textContent=matches.length?matches.length+'개 문서를 찾았습니다.':'일치하는 문서가 없습니다. 영문 약어나 함수 이름으로도 찾아보세요.';
  for(const {d}of matches){let at=d.text.toLocaleLowerCase().indexOf(terms[0]);if(at<0)at=0;const excerpt=(at>55?'…':'')+d.text.slice(Math.max(0,at-55),at+155)+'…';const a=document.createElement('a');a.className='search-result';a.href=(document.body.dataset.root||'./')+d.url;a.innerHTML='<small>'+escape(d.section)+'</small><b>'+escape(d.title)+'</b><p>'+escape(excerpt)+'</p>';results.append(a);}
 }
 input?.addEventListener('input',search);
 const routes={
  prepared:{steps:['준비 단계에서 ATen 대상과 입력 조건을 확인','TCL·EDF를 준비하고 custom op 구현을 등록','복사한 FX 그래프의 target을 준비된 op로 치환','forward에서 저장된 NPU 실행 객체 호출'],note:'준비 단계와 실행 단계를 구분합니다. ATen 전체를 전역으로 교체하거나 호출 때마다 TCL을 컴파일하는 구조가 아닙니다. 10·11장의 코드 경로를 확인하세요.'},
  export:{steps:['export_model이 모델 그래프와 입력 조건을 수집','지원 영역을 나누고 TCL·EDF와 바인딩을 준비','모델 단계·텐서 연결·상태가 담긴 실행계획 생성','fxbX 실행기가 계획을 검증하고 장치 프로그램 호출'],note:'EDF가 존재하는 것과 모델의 전체 실행계획이 완성된 것은 별개의 조건입니다. 계획 스키마·SDK·topology가 맞아야 합니다. 11·12장을 참고하세요.'},
  llm:{steps:['지원 모델의 설정·가중치와 호환되는 FXB를 로드','서빙 스케줄러가 요청·KV cache·버킷을 선택','준비된 커널과 입력을 런타임에 전달','결과·sampling·다음 decode 단계를 처리'],note:'공식 Furiosa-LLM의 모델·FXB 계약입니다. 임의 fxbX 파일을 전달한다고 새 모델 아키텍처가 자동으로 등록되지는 않습니다. 03·14장을 참고하세요.'},
  opt:{steps:['Rust device 함수에서 vISA 계산·메모리 배치를 표현','cargo furiosa-opt가 표현을 장치 실행물로 컴파일','Device Function Image를 furiosa-opt-rt에 적재','버퍼를 바인딩하고 함수를 제출한 뒤 완료 확인'],note:'vISA는 중간 표현입니다. 공식 검증 문서는 컴파일된 EDF를 설명하며, opt 실행기는 Device Function Image를 받습니다. SDK EDF와 동일 포맷·ABI인지, 교차 적재할 수 있는지는 별도 검증이 필요합니다. 04·08장을 참고하세요.'}
 };
 for(const lab of $$('[data-lab]')){
  const type=lab.dataset.lab;
  const v=key=>Number($('[data-key="'+key+'"]',lab)?.value||0);
  const out=(key,value)=>{const n=$('[data-value="'+key+'"]',lab);if(n)n.textContent=value;};
  const stat=(key,value)=>{const n=$('[data-stat="'+key+'"]',lab);if(n)n.textContent=value;};
  const update=()=>{
   if(type==='bucket'){
    const n=v('tokens'),preset=$('[data-key=preset]',lab).value;
    const buckets={fine:[128,256,512,1024,2048],coarse:[512,2048],single:[2048]}[preset];
    const bucket=buckets.find(b=>b>=n);out('tokens',n.toLocaleString());stat('bucket',bucket.toLocaleString());stat('padding',(bucket-n).toLocaleString());stat('use',(100*n/bucket).toFixed(1)+'%');$('[data-bar=used]',lab).style.width=(100*n/bucket)+'%';$('.bucket-bar',lab).setAttribute('aria-label',`전체 ${bucket}토큰 중 실제 ${n}, 패딩 ${bucket-n}`);
   }
   if(type==='dma'){
    const copy=v('copy'),compute=v('compute'),tiles=v('tiles'),serial=tiles*(copy+compute),overlap=copy+compute+(tiles-1)*Math.max(copy,compute);
    out('copy',copy+' ms');out('compute',compute+' ms');out('tiles',tiles);stat('serial',serial+' ms');stat('overlap',overlap+' ms');stat('saving',(100*(1-overlap/serial)).toFixed(1)+'%');
    const w=620,pad=70,scale=(w-pad-15)/serial;
    let svg=`<svg viewBox="0 0 ${w} 168" role="img" aria-label="순차 실행 ${serial}ms, 복사와 계산 겹침 ${overlap}ms"><text x="0" y="25" font-size="11" fill="#526259">순차</text><text x="0" y="83" font-size="11" fill="#526259">DMA</text><text x="0" y="121" font-size="11" fill="#526259">계산</text>`;
    const rect=(time,y,len,fill,label)=>`<rect x="${pad+time*scale}" y="${y}" width="${Math.max(2,len*scale-1)}" height="25" rx="3" fill="${fill}"/><text x="${pad+(time+len/2)*scale}" y="${y+17}" text-anchor="middle" font-size="9" fill="white">${label}</text>`;
    for(let i=0;i<tiles;i++){
     svg+=rect(i*(copy+compute),8,copy,'#829a8c','D'+(i+1))+rect(i*(copy+compute)+copy,8,compute,'#2f6a5c','C'+(i+1));
     const start=i*Math.max(copy,compute);svg+=rect(start,65,copy,'#829a8c','D'+(i+1))+rect(start+copy,103,compute,'#2f6a5c','C'+(i+1));
    }
    svg+=`<text x="${pad}" y="155" font-size="10" fill="#526259">0 ms</text><text x="${w-15}" y="155" text-anchor="end" font-size="10" fill="#526259">${serial} ms</text></svg>`;$('[data-timeline]',lab).innerHTML=svg;
   }
   if(type==='optimization'){
    const fraction=v('fraction')/100,factor=v('factor'),after=(1-fraction)+fraction/factor;out('fraction',Math.round(fraction*100)+'%');out('factor',factor+'×');stat('after',(100*after).toFixed(1));stat('speedup',(1/after).toFixed(2)+'×');
   }
  };
  $$('input,select',lab).forEach(control=>control.addEventListener('input',update));update();
  if(type==='route'){
   const choose=button=>{$$('[data-route]',lab).forEach(b=>b.setAttribute('aria-selected',String(b===button)));const route=routes[button.dataset.route];$('[data-route-steps]',lab).innerHTML=route.steps.map((text,i)=>`<li><b>0${i+1}</b><span>${escape(text)}</span></li>`).join('');$('[data-route-note]',lab).textContent=route.note;};
   $$('[data-route]',lab).forEach(b=>b.addEventListener('click',()=>choose(b)));choose($('[data-route]',lab));
  }
 }
})();
