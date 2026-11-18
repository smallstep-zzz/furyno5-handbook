/* Keep common UI labels in the language developers actually see in tools. */
const labels={
  '시작하기':'Getting Started',
  '전체 지도':'Overview',
  '실행 경로':'Execution Paths',
  '관측 & 성능':'Observability & Performance',
  '용어 사전':'Glossary',
  '다음 장':'Next',
  '이 책의 사용법':'How to use this guide',
  '권장 순서':'Recommended path',
  '실습 환경 체크리스트':'Environment checklist',
  '먼저 읽을 주의사항':'Before you start',
  '비공식 · 실습 중심 안내서':'Unofficial · Hands-on Guide',
  '문서 내 검색은 브라우저 찾기 사용':'Press ⌘K to search this page',
  '설치된 SDK가 곧 현실이다':'Start with the installed SDK',
  '언제 NPU에서 의미가 생기나':'When low-level work pays off on NPU'
};
const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);
nodes.forEach(node=>{let value=node.nodeValue;for(const [from,to] of Object.entries(labels))value=value.split(from).join(to);if(value!==node.nodeValue)node.nodeValue=value});
document.querySelectorAll('[data-copy]').forEach(btn=>{btn.addEventListener('click',async()=>{const code=btn.closest('.copy-wrap')?.querySelector('code')?.innerText;if(code){await navigator.clipboard.writeText(code);btn.textContent='Copied';setTimeout(()=>btn.textContent='Copy',1200)}})});
