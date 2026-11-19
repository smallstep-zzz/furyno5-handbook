import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
const require=createRequire(import.meta.url);
let Marked;
try { ({Marked}=require('marked')); }
catch { ({Marked}=require(path.join(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES || '', 'marked'))); }
const root=path.resolve(path.dirname(new URL(import.meta.url).pathname),'..');
const chapters=JSON.parse(fs.readFileSync(path.join(root,'data/chapters.json'),'utf8'));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const plain=s=>s.replace(/<[^>]+>/g,' ').replace(/&[a-z#0-9]+;/gi,' ').replace(/\s+/g,' ').trim();
const outPath=c=>c.output||`chapter/${c.id}.html`;
const outputFiles=[];
const search=[];
const allSources=new Map();
let diagrams=0,codes=0;

function lab(type){
 const start=`<section class="lab" data-lab="${esc(type)}"><div class="lab-top"><span class="lab-tag">INTERACTIVE · 설명용 모델</span></div>`;
 if(type==='bucket')return start+`<h3>버킷을 바꾸면 얼마나 남을까?</h3><div class="lab-controls"><label>실제 입력 길이 <output data-value="tokens">120</output><input aria-label="실제 입력 토큰 수" data-key="tokens" type="range" min="1" max="2048" step="1" value="120"></label><label>준비한 버킷<select aria-label="버킷 집합 선택" data-key="preset"><option value="fine">촘촘한 집합 · 128, 256, 512, 1024, 2048</option><option value="coarse">성긴 집합 · 512, 2048</option><option value="single">큰 버킷 하나 · 2048</option></select></label></div><div class="stat-row"><div class="stat"><small>선택한 버킷</small><b data-stat="bucket">128</b></div><div class="stat"><small>패딩 토큰</small><b data-stat="padding">8</b></div><div class="stat"><small>유효 길이 비율</small><b data-stat="use">93.8%</b></div></div><div class="bucket-bar" role="img" aria-label="유효 토큰과 패딩 토큰의 비율"><span data-bar="used" style="width:93.75%"></span></div><div class="bar-legend"><span>초록: 실제 입력</span><span>회색: 패딩</span></div><p class="explain">길이 한 축만 비교한 개념 모델입니다. 실제 attention 버킷은 배치·문맥·새 입력 길이를 함께 다루며, 패딩량만으로 실행시간을 예측할 수 없습니다. 가장 작은 호환 버킷을 선택하는 교육용 규칙입니다.</p></section>`;
 if(type==='dma')return start+`<h3>복사와 계산을 겹쳐 보기</h3><div class="lab-controls"><label>타일 하나의 복사 시간 <output data-value="copy">3 ms</output><input aria-label="타일 복사 시간" data-key="copy" type="range" min="1" max="10" value="3"></label><label>타일 하나의 계산 시간 <output data-value="compute">5 ms</output><input aria-label="타일 계산 시간" data-key="compute" type="range" min="1" max="10" value="5"></label><label>타일 수 <output data-value="tiles">4</output><input aria-label="타일 수" data-key="tiles" type="range" min="2" max="8" value="4"></label></div><div class="stat-row"><div class="stat"><small>완전히 직렬</small><b data-stat="serial">32 ms</b></div><div class="stat"><small>이상적인 겹침</small><b data-stat="overlap">23 ms</b></div><div class="stat"><small>계산상 단축</small><b data-stat="saving">28.1%</b></div></div><div class="timeline" data-timeline></div><p class="explain">독립적인 복사 엔진과 계산 엔진, 별도 버퍼, 타일 간 계산 독립성을 가정합니다. 시작·마무리 비용은 포함하지만 제출·동기화·결과 회수·자원 경합은 생략한 설명용 모델입니다. RNGD의 측정값이 아닙니다.</p></section>`;
 if(type==='optimization')return start+`<h3>어느 구간부터 줄일까?</h3><div class="lab-controls"><label>전체 시간 중 커널 계산의 비중 <output data-value="fraction">30%</output><input aria-label="커널 계산 비중" data-key="fraction" type="range" min="5" max="95" step="5" value="30"></label><label>커널만 빨라지는 배수 <output data-value="factor">2×</output><input aria-label="커널 개선 배수" data-key="factor" type="range" min="1" max="10" value="2"></label></div><div class="stat-row"><div class="stat"><small>기존 전체 시간</small><b>100</b></div><div class="stat"><small>개선 후 전체 시간</small><b data-stat="after">85.0</b></div><div class="stat"><small>전체 속도 향상</small><b data-stat="speedup">1.18×</b></div></div><p class="explain">직렬 구간에 대한 Amdahl의 법칙: 개선 후 시간 = (1 − 계산 비중) + 계산 비중 / 개선 배수. 단위 없는 상대시간이며, CPU 대기·복사·상태 갱신이 그대로일 때의 계산입니다. 겹치는 실행에는 구간 합산 대신 실제 임계경로를 측정하세요.</p></section>`;
 if(type==='route')return start+`<h3>어느 실행 경로를 보고 있나요?</h3><div class="tabs" role="tablist" aria-label="실행 경로 선택"><button type="button" role="tab" aria-selected="true" data-route="prepared">FuryNo5 prepared</button><button type="button" role="tab" aria-selected="false" data-route="export">FuryNo5 export</button><button type="button" role="tab" aria-selected="false" data-route="llm">공식 Furiosa-LLM</button><button type="button" role="tab" aria-selected="false" data-route="opt">furiosa-opt</button></div><ol class="route-steps" data-route-steps></ol><p class="explain" data-route-note>같은 NPU를 사용해도 모델 실행계획을 만들고 요청을 스케줄링하는 주체가 다릅니다.</p></section>`;
 return `<p class="callout">이 개념의 조절 실험은 <a href="06-buckets-shapes.html">버킷 설계</a>와 <a href="07-optimization.html">최적화</a> 장에서 볼 수 있습니다.</p>`;
}

function page(c,index,raw){
 const output=outPath(c), prefix=output.includes('/')?'../':'';
 const headings=[],ids=new Set(); let diagramNo=0;
 const isHome=index===0;
 raw=raw.replace(/^---\n[\s\S]*?\n---\n/,'');
 raw=raw.replace(/^#\s+.*\r?\n/,'');
 raw=raw.replace(/<!--\s*interactive:([\w-]+)\s*-->/g,(_,type)=>lab(type));
 raw=raw.replace(/\]\((?:\.\/)?([\w-]+)\.md(#[^)]*)?\)/g,(_,id,hash)=>{const target=chapters.find(x=>x.id===id);return target?`](${prefix}${outPath(target)}${hash||''})`:`](${id}.md${hash||''})`;});
 const renderer={
  heading({tokens,depth,text}){
   const rendered=this.parser.parseInline(tokens);let id=plain(rendered).toLowerCase().replace(/[^\p{L}\p{N}]+/gu,'-').replace(/^-|-$/g,'');
   const base=id;let k=2;while(ids.has(id))id=base+'-'+k++;ids.add(id);
   if(depth<=3)headings.push({id,title:plain(rendered),depth});
   return `<h${depth} id="${esc(id)}"><a class="heading-anchor" href="#${esc(id)}">${rendered}</a></h${depth}>\n`;
  },
  code({text,lang}){
   if(lang==='mermaid'){
    diagramNo++;diagrams++;
    const hash=crypto.createHash('sha256').update(text).digest('hex').slice(0,12);
    const file=`assets/diagrams/${c.id}-${hash}`;
    const mmd=path.join(root,file+'.mmd'), svg=path.join(root,file+'.svg');
    fs.writeFileSync(mmd,text+'\n');
    if(!fs.existsSync(svg)||process.argv.includes('--render')){
     execFileSync('python3',[path.join(root,'tools/render_diagrams.py'),mmd,svg],{stdio:'pipe'});
    }
    return `<figure class="diagram"><img src="${prefix}${file}.svg" alt="${esc(c.title)} · 도식 ${diagramNo}" loading="lazy"><figcaption>도식 ${diagramNo} · ${esc(c.title)} — 설명을 위해 단순화한 관계도. 실제 호출 조건과 근거는 본문을 함께 읽으세요.</figcaption><details><summary>그림의 구조 보기</summary><pre>${esc(text)}</pre></details></figure>`;
   }
   codes++;
   return `<div class="codebox"><div class="codebar"><span>${esc(lang||'text')}</span><button type="button" class="copy-code" aria-label="코드 복사">복사</button></div><pre><code class="language-${esc(lang||'text')}">${esc(text)}\n</code></pre></div>\n`;
  },
  table(token){return '<div class="table-wrap">'+this.constructor.prototype.table.call(this,token)+'</div>';}
 };
 const parser=new Marked({gfm:true,breaks:false,renderer});
 let body=parser.parse(raw);
 body=body.replace(/<a href="(https:\/\/github\.com\/smallstep-zzz\/(?:FuryNo5|Furiosa-Hug-Anything|FuryBoost)\/[^"#]*)(#[^"]*)?">([\s\S]*?)<\/a>/g,(m,url,hash,label)=>`<a href="${url}${hash||''}">${label}</a><span class="source-private" title="이 소스 저장소는 조회일 기준 비공개입니다. 접근 권한이 필요합니다.">권한 필요</span>`);
 const chapterText=plain(body);
 const readTime=Math.max(4,Math.round(chapterText.length/800));
 const toc=headings.map(h=>`<a class="${h.depth===3?'minor':''}" href="#${esc(h.id)}">${esc(h.title)}</a>`).join('');
 let nav='',part='';
 for(const n of chapters){if(n.part!==part){part=n.part;nav+=`<div class="nav-caption">${esc(part)}</div>`;}nav+=`<a class="nav-link${n.id===c.id?' active':''}" ${n.id===c.id?'aria-current="page"':''} href="${prefix}${outPath(n)}"><span class="num">${/^\d/.test(n.id)?n.id.slice(0,2):'·'}</span><span>${esc(n.short)}</span></a>`;}
 const prev=chapters[index-1],next=chapters[index+1];
 const pager=`<nav class="next-prev" aria-label="이전과 다음 장">${prev?`<a href="${prefix}${outPath(prev)}"><small>← 이전 장</small><b>${esc(prev.title)}</b></a>`:'<span></span>'}${next?`<a href="${prefix}${outPath(next)}"><small>다음 장 →</small><b>${esc(next.title)}</b></a>`:'<span></span>'}</nav>`;
 const title=isHome?'하드웨어에서 모델 서빙까지,<br>FuryNo5 개발 핸드북':esc(c.title);
 const badges=`<div class="badge-row"><span class="badge"><i></i>공식 자료</span><span class="badge"><i></i>코드 확인</span><span class="badge opinion"><i></i>FuryNo5의 해석</span><span class="badge unknown"><i></i>미확인·개발 과제</span></div>`;
 const cards=isHome?`<div class="book-cards">${[chapters[1],chapters[4],chapters[10],chapters[13]].map((x,i)=>`<a class="book-card" href="${outPath(x)}"><small>PART ${['I','II','III','IV'][i]}</small><b>${['디바이스를 이해하기','프로그램을 만들고 실행하기','FuryNo5를 해부하기','모델 포팅과 서빙'][i]}</b><p>${esc(x.summary)}</p><span class="arrow" aria-hidden="true">↗</span></a>`).join('')}</div>`:'';
 const html=`<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="${esc(c.summary)}"><meta name="theme-color" content="#fbfaf7"><title>${esc(c.title)} · FuryNo5 개발 핸드북</title><link rel="icon" href="${prefix}assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="${prefix}assets/site.css"><script defer src="${prefix}assets/site.js"></script></head>
<body data-root="${prefix||'./'}" data-chapter="${esc(c.id)}"><a class="skip" href="#content">본문으로 이동</a><header class="topbar"><button class="menu-toggle" type="button" aria-label="전체 목차 열기" aria-controls="sidebar" aria-expanded="false">☰</button><a class="brand" href="${prefix}index.html"><span class="brand-mark">Nº5</span>FuryNo5<small>DEVELOPER HANDBOOK</small></a><div class="top-actions"><a class="top-link" href="${prefix}sources.html">출처와 버전</a><button class="search-open" type="button" aria-label="핸드북 전체 검색"><span>전체 문서 검색</span><kbd>⌘ K</kbd></button></div></header><div class="progress" aria-hidden="true"></div><div class="mobile-shade"></div><aside class="sidebar" id="sidebar" aria-label="전체 목차"><nav>${nav}</nav><div class="side-foot">비공식 개발 핸드북 · 2026.11.19<br>공식 SDK와 프로젝트 구현을 구분합니다.<br><a href="https://github.com/smallstep-zzz/furyno5-handbook">원고와 변경 이력 ↗</a></div></aside><div class="shell"><main class="page" id="content"><div class="eyebrow">${isHome?'THE PATH FROM SILICON TO SERVING':esc(c.part)+' / '+(/^\d/.test(c.id)?'CHAPTER '+c.id.slice(0,2):'REFERENCE')}</div><h1>${title}</h1><p class="dek">${esc(c.summary)}</p>${isHome?badges:''}<div class="page-meta"><span>검토 2026.11.19</span><span>SDK 2026.4 계열 / 버전별 표기</span><span class="readtime">약 ${readTime}분</span><span><a href="https://github.com/smallstep-zzz/furyno5-handbook/blob/main/content/${c.id}.md">원고</a></span></div>${cards}<details class="mobile-toc"><summary>이 장의 목차</summary><div>${toc}</div></details><noscript><p class="noscript">본문과 그림은 JavaScript 없이 읽을 수 있습니다. 검색과 조절 실험에는 JavaScript가 필요합니다.</p></noscript><article class="article">${body}</article>${pager}<footer class="footer">FuryNo5 개발 핸드북 · 비공식 자료<br>공식 문서와 소스의 권리는 각 프로젝트에 있습니다. 코드 인용은 출처와 함께 표시합니다.<br><a href="${prefix}review.html">검토 기록과 남은 질문</a> · <a href="${prefix}sources.html">출처와 버전</a> · <a href="${prefix}glossary.html">용어 사전</a></footer></main></div><aside class="right-toc" aria-label="이 장의 목차"><h2>이 장에서 다루는 내용</h2>${toc}</aside><dialog class="search-dialog" aria-label="문서 검색"><div class="search-head"><input type="search" aria-label="검색어" placeholder="DMA, EDF, 폴백, 버킷…" autocomplete="off"><button class="close-search" type="button">닫기</button></div><div class="search-status" role="status">용어나 함수 이름을 입력하세요.</div><div class="search-results"></div></dialog></body></html>`;
 fs.mkdirSync(path.dirname(path.join(root,output)),{recursive:true});fs.writeFileSync(path.join(root,output),html);outputFiles.push(output);
 search.push({title:c.title,section:c.part,url:output,text:chapterText});
 for(const match of raw.matchAll(/\]\((https?:\/\/[^\s)]+)(?:\s+"[^"]*")?\)/g)){
   const url=match[1]; if(!allSources.has(url))allSources.set(url,new Set());allSources.get(url).add(c.id);
 }
}

for(let i=0;i<chapters.length;i++){
 const c=chapters[i], f=path.join(root,'content',c.id+'.md');
 if(!fs.existsSync(f)){if(process.argv.includes('--partial'))continue;throw new Error('Missing chapter: '+f);}
 page(c,i,fs.readFileSync(f,'utf8'));
}
fs.writeFileSync(path.join(root,'search-index.json'),JSON.stringify(search));outputFiles.push('search-index.json');
const sourceData=[...allSources.entries()].map(([url,ids])=>({url,chapters:[...ids],checked_on:'2026-11-19',access:/github.com\/smallstep-zzz\/(FuryNo5|Furiosa-Hug-Anything|FuryBoost)\//.test(url)?'repository permission required':'public'}));
fs.writeFileSync(path.join(root,'data/source-index.json'),JSON.stringify(sourceData,null,2)+'\n');
const aliases={'01-map':'../index.html','02-stack-hardware':'01-hardware.html','03-aten-tcl':'09-torch-aten.html','04-tcl-hands-on':'04-tcl-edf.html','05-eisa-visa':'08-eisa-visa.html','06-execution-paths':'05-graphs-fallback.html','07-llm-vllm':'14-serving-vllm.html','08-smi-performance':'17-debugging.html'};
for(const [name,target]of Object.entries(aliases)){
 const f=`chapter/${name}.html`;fs.writeFileSync(path.join(root,f),`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=${target}"><title>새 핸드북으로 이동</title><link rel="canonical" href="${target}"></head><body><p>개편된 <a href="${target}">FuryNo5 개발 핸드북</a>으로 이동합니다.</p></body></html>`);outputFiles.push(f);
}
fs.writeFileSync(path.join(root,'data/build-report.json'),JSON.stringify({chapters:search.length,diagrams,code_blocks:codes,source_urls:sourceData.length,generated_pages:outputFiles.filter(x=>x.endsWith('.html')),verification:'Static document build; no NPU hardware execution performed.'},null,2)+'\n');
console.log(JSON.stringify({chapters:search.length,diagrams,code_blocks:codes,source_urls:sourceData.length}));
