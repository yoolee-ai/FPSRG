import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const samples=JSON.parse(fs.readFileSync(path.join(root,'demo_sample.json'),'utf8'));
const embedded=html.match(/<script id="sample-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const code=scripts[scripts.length-1][1];
new vm.Script(code,{filename:'index.html#application'});
assert.deepEqual(JSON.parse(embedded),samples,'embedded and downloaded samples must agree');
assert.equal(samples.synthetic,true);
assert.equal(samples.people.length,8);
assert.equal(new Set(samples.people.map(p=>p.id)).size,8);
assert.equal(samples.people.reduce((n,p)=>n+p.dialogue.length,0),56);
const suggestions=[...html.matchAll(/<button type="button" class="suggestion">([^<]*)<\/button>/g)].map(match=>match[1]);
assert.deepEqual(suggestions,[
  '买了不吃亏，买了不上当，今天这个价很划算。',
  '这个优惠只到今天，现在不下单，明天就恢复原价了。',
  '大家都选这款，档次高又有面子，你也值得买个好的。',
  '这款升级版功能更多，现在一步到位，后面就不用再换了。',
  '你这都不买，是不是没钱',
  '对不起，我错了',
  '你傻逼吧，不买就别问',
],'all seven requested shortcuts are shown verbatim');
assert(!html.includes('穷就别问了')&&!html.includes('赶紧滚'));
assert(!html.includes('每轴一次调用 · 对所有人物分别判断 · 无自动群聊'));
for(const p of samples.people){
  assert(p.name&&p.dialogue.length>=5);
  assert(!('scores' in p)&&!('levels' in p)&&!('matched' in p),'no invented extraction outcomes');
  for(const turn of p.dialogue){assert(['customer','staff'].includes(turn.role));assert.equal(typeof turn.text,'string');assert(turn.text.length>10);}
}

class Element{
  constructor(tag='div'){this.tagName=tag;this.children=[];this.style={};this.attributes={};this.textContent='';this.value='';this.hidden=false;this.disabled=false;this.className='';this.listeners={};this.scrollTop=0;this.scrollHeight=1000;this.clientHeight=300;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=[...children];}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(name,handler){this.listeners[name]=handler;}
  focus(){this.focused=true;}
}
const nodes=new Map();
const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
get('sample-data').textContent=embedded;
const shortcutButtons=suggestions.map(text=>{const node=new Element('button');node.textContent=text;return node;});
const emotionButtons=[...html.matchAll(/<button type="button" class="emotion-button" data-emotion="([a-z]+)" aria-pressed="(true|false)">([^<]*)<\/button>/g)].map(([,emotion,pressed,label])=>{const node=new Element('button');node.dataset={emotion};node.textContent=label;node.setAttribute('aria-pressed',pressed);return node;});
assert.deepEqual(emotionButtons.map(button=>[button.dataset.emotion,button.textContent]),[['joy','增加开心'],['anger','增加愤怒'],['none','关闭情绪']]);
assert.deepEqual(emotionButtons.filter(button=>button.attributes['aria-pressed']==='true').map(button=>button.dataset.emotion),['none']);
assert(html.includes('从下一条回复生效；保持至关闭；仅实验控制，未重新标定人格分数'));
const document={getElementById:get,createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element('#fragment'),createTextNode:text=>({textContent:String(text)}),querySelectorAll:selector=>selector==='.suggestion'?shortcutButtons:selector==='.emotion-button'?emotionButtons:[]};
const sandbox={document,URL,location:{pathname:'/',href:'http://localhost/'},console,setTimeout:()=>0,clearTimeout:()=>{}};
const ctx=vm.createContext(sandbox);
const bindStart=code.indexOf("    $('extract-button').addEventListener");
assert(bindStart>0);
vm.runInContext(code.slice(0,bindStart),ctx);
const run=script=>vm.runInContext(script,ctx);
const walk=(node,predicate)=>[...(predicate(node)?[node]:[]),...(node.children||[]).flatMap(child=>walk(child,predicate))];
const allText=node=>[node.textContent,...(node.children||[]).map(allText)].filter(Boolean).join(' ');

assert.equal(run('AXES.map(a=>a.key).join("")'),'ACNRMT');
assert.equal(run('AXES.length'),6);
run('loadSample()');assert.equal(JSON.parse(get('json-input').value).people.length,8);
assert.equal(run('parseInput().people.length'),8);
assert.match(get('input-summary').textContent,/8 位人物 · 56 条对话/);
run('setProgress(3)');assert.equal(get('progress-count').textContent,'3 / 6');assert.equal(get('progress-fill').style.width,'50%');
run('setProgress(99)');assert.equal(get('progress-count').textContent,'6 / 6');assert.equal(get('progress-fill').style.width,'100%');
run('setProgress(-2)');assert.equal(get('progress-count').textContent,'0 / 6');

ctx.person={id:'p1',name:'<b>虚构人物</b>',levels:{A:'low',C:'high',N:'low',R:'high',M:'medium',T:'high'},evidence:{A:'证据不应变为HTML'},confidence:{A:'high'},matched:{node_id:123,scores:{A:41.15,C:91.41,N:0,R:61.57,M:50,T:73.83},level_mismatches:2}};
run('renderProfile(person)');
assert.equal(get('profile-name').textContent,'<b>虚构人物</b>','render text literally');
const rows=walk(get('profile-axes'),n=>n.className==='axis-row');assert.equal(rows.length,6);
assert.equal(walk(get('profile-axes'),n=>n.className==='axis-anchor').length,0,'original-score markers are not rendered');
const bars=walk(get('profile-axes'),n=>n.className==='axis-bar');assert.equal(bars.length,6);
assert.deepEqual(bars.map(bar=>bar.style.width),['20%','80%','20%','80%','50%','80%']);
assert(!/原测量|原格点测量|41\.1|91\.4|61\.5|73\.8/.test(allText(get('profile-axes'))),'profile shows only inferred target levels, not original scores');
for(const node of walk(get('profile-axes'),()=>true)){
  assert(!/原测量|原格点测量|41\.1|91\.4|61\.5|73\.8/.test(JSON.stringify(node.attributes||{})),'accessibility text must also omit original scores');
  assert(!/原测量|原格点测量|41\.1|91\.4|61\.5|73\.8/.test(node.title||''),'tooltips must also omit original scores');
}
assert.match(get('mapping-note').textContent,/补充后未复测/);
assert.match(get('mapping-note').textContent,/目标档位，不是实测分数/);
assert.equal(ctx.person.matched.scores.A,41.15,'display simplification does not overwrite measured catalog scores');
const displayedBefore=allText(get('profile-axes'));
ctx.person.matched.scores.A=undefined;run('renderProfile(person)');
assert.equal(walk(get('profile-axes'),n=>n.className==='axis-anchor').length,0);
assert.equal(allText(get('profile-axes')),displayedBefore,'target display does not depend on hidden original-score availability');

assert.throws(()=>run("installExtraction({individuals:[{id:'old',name:'old',matched:{node_id:1},levels:{A:'low',C:'high',N:'low',R:'high'}}]})"),/提取结果结构不完整/);
assert.equal(run("historyMessages([{role:'system',content:'x'},{role:'user',content:'hello'},{role:'assistant',content:'yes'}]).length"),2);
assert.equal(run("scoreLevel(100/3)"),'medium');assert.equal(run("scoreLevel(200/3)"),'high');

assert.equal(run("savedMode('p1')"),'shared');assert.equal(run("normalizedMode(undefined)"),'shared');
assert(html.includes('data-mode="shared"')&&!html.includes('data-mode="continue"'));
run(`state.people=[person,{...person,id:'p2',name:'另一人物'}];state.selected='p1';state.status={workers_ready:8,loading:false};
  for(const id of ['p1','p2'])state.chats.set(id,{mode:'shared',threads:{shared:{messages:[],draft:''},none:{messages:[],draft:''}},pending:false,pendingMode:null,resetting:false,reconnecting:false,progress:null,progressThoughtOpen:true,contexts:{},contextLoading:{},contextVersions:{},scrolls:{}});
  renderMessages();`);
assert.equal(get('suggestions').hidden,false,'shortcuts remain visible before first turn');
assert(shortcutButtons.every(button=>!button.disabled));
const shortcutBinding=code.split('\n').find(line=>line.includes("document.querySelectorAll('.suggestion').forEach(button=>button.addEventListener"));
run(shortcutBinding);
shortcutButtons[3].listeners.click();assert.equal(get('chat-input').value,suggestions[3]);
shortcutButtons[4].listeners.click();assert.equal(get('chat-input').value,suggestions[4]);
assert.equal(run("state.chats.get('p1').threads.shared.draft"),suggestions[4]);
for(const index of [5,6]){
  shortcutButtons[index].listeners.click();
  assert.equal(get('chat-input').value,suggestions[index]);
  assert.equal(run("state.chats.get('p1').threads.shared.draft"),suggestions[index]);
  assert.equal(run("state.chats.get('p1').threads.shared.messages.length"),0,'new shortcuts only fill, never send');
}
shortcutButtons[3].listeners.click();
assert.equal(run("state.chats.get('p1').threads.shared.messages.length"),0,'shortcut only fills, never sends');
assert.equal(run("state.chats.get('p1').threads.shared.draft"),suggestions[3]);
run("state.chats.get('p1').threads.shared.messages.push({role:'user',answer:'统一问题'});state.chats.get('p1').pending=true;state.chats.get('p1').pendingMode='shared';renderMessages();");
assert.equal(get('suggestions').hidden,false,'shortcuts remain visible after a question');assert(shortcutButtons.every(button=>button.disabled),'shortcuts disabled while current person generates');
const firstView=run("state.chats.get('p1').liveView"),firstChildren=get('chat-messages').children;
get('chat-messages').scrollTop=100;
run("receiveChatProgress('p1',{streaming:true,tokens:3,seconds:1.1,reasoning:'<script>真实片段</script>',answer:'',raw_text:'<script>真实片段</script>',reasoning_complete:false});");
assert.equal(firstView.thoughtText.textContent,'<script>真实片段</script>');assert.equal(firstView.thought.open,true);
assert.equal(firstView.thought.hidden,false);assert.match(firstView.thoughtLabel.textContent,/生成中/);
assert.equal(get('chat-messages').children,firstChildren,'incremental output does not rebuild conversation DOM');
assert.equal(get('chat-messages').scrollTop,100,'reading older content does not scroll to bottom');
assert.equal(run("state.chats.get('p1').threads.shared.messages.length"),1,'partial thoughts do not enter history');
firstView.thought.open=false;firstView.thought.listeners.toggle();
run("receiveChatProgress('p1',{streaming:true,tokens:8,seconds:2,reasoning:'新的真实思考',answer:'回答片段',raw_text:'原始片段',reasoning_complete:true});");
assert.equal(firstView.thought.open,false,'polls preserve user fold state');assert.equal(firstView.answerText.textContent,'回答片段');assert.equal(firstView.answer.hidden,false);
run("state.chats.get('p1').reconnecting=true;updateStreamingView('p1');");assert.match(firstView.status.textContent,/连接中断/);assert.equal(firstView.thoughtText.textContent,'新的真实思考');
run("state.selected='p2';renderMessages();receiveChatProgress('p1',{streaming:true,tokens:10,seconds:3,reasoning:'离开后仍保存的思考',answer:'回答片段二',raw_text:'原文',reasoning_complete:true});");
assert(!allText(get('chat-messages')).includes('离开后仍保存的思考'),'other person never sees current person partial');
assert.equal(run("state.chats.get('p2').threads.shared.messages.length"),0);
run("state.selected='p1';renderMessages();");assert.equal(run("state.chats.get('p1').liveView.thoughtText.textContent"),'离开后仍保存的思考');assert.equal(run("state.chats.get('p1').liveView.thought.open"),false,'switching people preserves thought folding');
run("const failed=retainedPartial(state.chats.get('p1'));state.chats.get('p1').threads.shared.messages.push({...failed,error:'网络任务终止'});state.chats.get('p1').pending=false;state.chats.get('p1').progress=null;renderMessages();");
assert.match(allText(get('chat-messages')),/离开后仍保存的思考/);assert.match(allText(get('chat-messages')),/未加入上下文/);assert.equal(get('suggestions').hidden,false);assert(shortcutButtons.every(button=>!button.disabled));
const common={context_mode:'shared',source_material_included:true,source_material:JSON.stringify({dialogue:[{role:'staff',text:'这款产品基础功能齐全，售价2000元，升级版售价3000元。'},{role:'customer',text:'我再考虑一下。'}]}),history:[],history_turns:0,notice:'相同预设对话，各自历史'};
ctx.common=common;
run("state.chats.get('p1').contexts.shared=common;renderContext('p1','shared');");
const commonText=allText(get('context-content'));
assert.match(commonText,/统一前置对话/);assert.match(commonText,/原始人物对话仅用于提取/);assert.match(commonText,/预设对话，不是本轮模型生成/);assert(!commonText.includes('真实片段'));
assert.deepEqual(walk(get('context-content'),node=>node.className==='context-speaker').map(node=>node.textContent),['销售','客户']);
assert.deepEqual(walk(get('context-content'),node=>node.className==='context-text').map(node=>node.textContent),['这款产品基础功能齐全，售价2000元，升级版售价3000元。','我再考虑一下。']);
run("state.chats.get('p2').contexts.shared=common;state.selected='p2';renderContext('p2','shared');");assert.equal(allText(get('context-content')),commonText,'all people display same preset dialogue with sales/customer speakers');
run("state.chats.get('p2').mode='none';state.chats.get('p2').contexts.none={context_mode:'none',source_material_included:false,source_material:'',history:[],history_turns:0,notice:'无上下文'};renderContext('p2','none');");
assert(!allText(get('context-content')).includes('售价2000元')&&!allText(get('context-content')).includes('我再考虑一下'),'no-context display excludes preset dialogue');

// A catalog rematch changes only the active configuration, not evidence, drafts, modes or past replies.
run("state.selected='p1';state.chats.get('p1').pending=true;state.chats.get('p1').progress={streaming:true,reasoning:'正在生成的原始片段',answer:'',tokens:4,seconds:1};state.chats.get('p1').threads.shared.draft='保留我的草稿';state.chats.get('p1').threads.shared.messages.push({role:'assistant',complete:true,answer:'旧节点实际回复',node_id:329});renderMessages();");
const beforeRematch=run("JSON.stringify({threads:state.chats.get('p1').threads,mode:state.chats.get('p1').mode,progress:state.chats.get('p1').progress,levels:state.people[0].levels,evidence:state.people[0].evidence})");
const rematchView=run("state.chats.get('p1').liveView"),rematchDOM=get('chat-messages').children;
ctx.latestMatched={...ctx.person.matched,node_id:416};
ctx.catalogRecord={id:'p1',matched:ctx.latestMatched,rematch_history:[{previous_node_id:329,node_id:416}],history:[{role:'assistant',content:'不应该从服务器覆盖当前页面历史'}],shared_history:[{role:'assistant',content:'同样不能覆盖'}],levels:{A:'high'}};
const requests=[];
ctx.fetch=async target=>{requests.push(target);return {ok:true,status:200,json:async()=>ctx.catalogRecord};};
await run("refreshPersonMatch('p1')");
assert.deepEqual(requests,['/api/people/p1'],'rematch refresh is read-only and never extracts or sends');
assert.equal(run("state.people[0].matched.node_id"),416);
assert.equal(run("state.people[0].rematch_history.length"),1);
assert.equal(run("JSON.stringify({threads:state.chats.get('p1').threads,mode:state.chats.get('p1').mode,progress:state.chats.get('p1').progress,levels:state.people[0].levels,evidence:state.people[0].evidence})"),beforeRematch);
assert.equal(run("state.chats.get('p1').liveView"),rematchView);assert.equal(get('chat-messages').children,rematchDOM,'rematch does not rebuild live conversation');
assert.match(get('profile-node').textContent,/#416/);assert.match(allText(get('chat-messages')),/本次网格 #329/,'old reply retains actual node');
run("state.selected='p2';renderProfile(state.people[1]);");const otherProfile=get('profile-node').textContent;
let resolveRefresh;
ctx.fetch=target=>{requests.push(target);return new Promise(resolve=>{resolveRefresh=resolve;});};
const staleRefresh=run("refreshPersonMatch('p1')");
ctx.dispatchedMatched={...ctx.person.matched,node_id:173};
run("applyDispatchedMatch('p1',dispatchedMatched)");
resolveRefresh({ok:true,status:200,json:async()=>({...ctx.catalogRecord,matched:{...ctx.latestMatched,node_id:329}})});
await staleRefresh;
assert.equal(run("state.people[0].matched.node_id"),173,'older GET does not undo dispatched matching');
assert.equal(get('profile-node').textContent,otherProfile,'background matching does not overwrite selected other-person profile');
assert.equal(run("state.chats.get('p1').dispatchedMatch.node_id"),173);
assert.equal(run("state.chats.get('p2').dispatchedMatch"),undefined,'actual dispatch snapshot belongs only to correct person');
run("state.selected='p1';renderMessages();");assert.match(run("state.chats.get('p1').liveView.meta.textContent"),/本次网格 #173/);
run("applyDispatchedMatch('p1',{...dispatchedMatched,node_id:329});state.chats.get('p1').threads.shared.messages.push({role:'assistant',complete:true,node_id:329,answer:'先前已提交任务的原始回复'});state.chats.get('p1').pending=false;renderMessages();");
ctx.fetch=async()=>({ok:true,status:200,json:async()=>({...ctx.catalogRecord,matched:{...ctx.latestMatched,node_id:416}})});
await run("refreshPersonMatch('p1')");
assert.equal(run("state.people[0].matched.node_id"),416,'finish-time GET restores newest rematch after old in-flight response');
assert.equal(run("state.chats.get('p1').threads.shared.messages.at(-1).node_id"),329,'past actual node is not relabeled after catalog refresh');
assert(code.includes('applyDispatchedMatch(id,ticket.matched)')&&code.includes('applyDispatchedMatch(id,result.matched)'));
assert(code.includes('updateControls();refreshPersonMatch(id);'),'terminal paths refresh latest configuration without changing their actual result snapshots');

// New entry/refresh never restores prior people or conversations, even if the URL contains an old extraction.
assert(!code.includes('localStorage'));
assert(!code.includes('savedExtraction'));
const startup=code.slice(code.indexOf('    // 每次打开或刷新'));
assert(startup.includes('loadSample();refreshStatus();'));
assert(!startup.includes('restoreExtraction('));
assert(!startup.includes('extractionToRestore'));
assert(code.includes("if(mode==='none'&&(data.source_material_included||data.source_material"),'no-context safety validation remains');
assert(code.includes('context_mode:mode'),'chat request binds selected mode');
assert(!code.includes('.innerHTML'),'untrusted conversation is rendered as text');
assert(html.includes('模型生成的思考'));
assert(html.includes('语义补充'));
assert(!html.includes('不额外添加人格提示词'));
assert(!html.includes('四轴')&&!html.includes('四次并行'));
assert(html.includes('98% 格式门控'));

// Emotion is a per-person control for future requests, independent of histories and modes.
ctx.emotionRequests=[];
run(`refreshContext=async()=>{};refreshPersonMatch=async()=>{};
  api=async(target,options)=>{emotionRequests.push({target,method:options?.method,body:options?.body?JSON.parse(options.body):null});
    if(target==='/api/chat')return {status_url:'/api/jobs/local_emotion_test'};
    if(options?.method==='DELETE')return {};
    throw new Error('Unexpected local test request: '+target);
  };
  pollTask=(target,onUpdate)=>new Promise(resolve=>{globalThis.finishEmotionReply=resolve;onUpdate?.({progress:{streaming:true,reasoning:'本地测试片段',answer:'',reasoning_complete:false}});});
  installExtraction({individuals:[{...person,emotion:'anger'},{...person,id:'p2',name:'另一人物',emotion:'joy'}]});
  state.status={workers_ready:8,loading:false};updateControls();`);
assert.equal(run("state.chats.get('p1').emotion"),'none','fresh extraction does not restore an old emotion');
assert.equal(run("state.chats.get('p2').emotion"),'none');
assert.equal(get('emotion-current').textContent,'当前：关闭情绪');
const emotionBinding=code.split('\n').find(line=>line.includes("document.querySelectorAll('.emotion-button').forEach(button=>button.addEventListener"));
assert(emotionBinding);run(emotionBinding);
const emotionButton=value=>emotionButtons.find(button=>button.dataset.emotion===value);
run(`state.chats.get('p1').threads.shared.messages.push({role:'assistant',answer:'必须保留的历史',complete:true});
  state.chats.get('p1').threads.none.messages.push({role:'user',answer:'独立模式历史'});
  state.chats.get('p1').threads.shared.draft='仍在编辑的草稿';$('chat-input').value='仍在编辑的草稿';renderMessages();`);
const beforeEmotion=run("JSON.stringify({people:state.people,threads:state.chats.get('p1').threads,mode:state.chats.get('p1').mode})");
const emotionDOM=get('chat-messages').children;
emotionButton('joy').listeners.click();
assert.equal(run("state.chats.get('p1').emotion"),'joy');
assert.equal(run("JSON.stringify({people:state.people,threads:state.chats.get('p1').threads,mode:state.chats.get('p1').mode})"),beforeEmotion,'switching emotion never edits personality, history or draft');
assert.equal(get('chat-messages').children,emotionDOM,'emotion controls do not rebuild conversation or disturb streaming DOM');
assert.equal(get('emotion-current').textContent,'当前：增加开心');
assert.equal(emotionButton('joy').attributes['aria-pressed'],'true');
assert.equal(emotionButtons.filter(button=>button.attributes['aria-pressed']==='true').length,1);
assert.equal(ctx.emotionRequests.length,0,'choosing an emotion is local and never sends a model request');
run("selectPerson('p2')");assert.equal(get('emotion-current').textContent,'当前：关闭情绪');
emotionButton('anger').listeners.click();
run("selectPerson('p1');switchMode('none')");
assert.equal(run("state.chats.get('p1').emotion"),'joy','context mode changes preserve this person\'s emotion');
assert.equal(run("state.chats.get('p2').emotion"),'anger','people have independent emotion choices');
assert.equal(run("state.chats.get('p1').threads.shared.messages[0].answer"),'必须保留的历史');
await run('resetChat()');
assert.equal(run("state.chats.get('p1').emotion"),'joy','clearing one history does not close a sustained emotion');
assert.equal(run("state.chats.get('p1').threads.none.messages.length"),0);
assert.equal(run("state.chats.get('p1').threads.shared.messages[0].answer"),'必须保留的历史');
for(const busy of ['resetting','pending']){
  run(`state.chats.get('p1').${busy}=true;updateControls();switchEmotion('anger');`);
  assert(emotionButtons.every(button=>button.disabled));
  assert.equal(run("state.chats.get('p1').emotion"),'joy','programmatic switches also respect busy guards');
  run(`state.chats.get('p1').${busy}=false;updateControls();`);
}
run("state.extracting=true;updateControls();switchEmotion('anger')");assert(emotionButtons.every(button=>button.disabled));
assert.equal(run("state.chats.get('p1').emotion"),'joy');
run("state.extracting=false;switchEmotion('invalid');updateControls()");assert.equal(run("state.chats.get('p1').emotion"),'joy');
run("$('chat-input').value='实际发送时采用已选情绪'");
const emotionSend=run('sendMessage()');
assert(emotionButtons.every(button=>button.disabled),'in-flight request locks emotion switching immediately');
assert.match(get('emotion-current').textContent,/增加开心.*切换已锁定/);
emotionButton('anger').listeners.click();
assert.equal(run("state.chats.get('p1').emotion"),'joy');
run("selectPerson('p2')");assert(emotionButtons.every(button=>!button.disabled),'one busy person does not lock another person\'s emotion');
assert.equal(get('emotion-current').textContent,'当前：增加愤怒');
emotionButton('none').listeners.click();run("selectPerson('p1')");
assert(emotionButtons.every(button=>button.disabled));
await new Promise(setImmediate);
const firstEmotionPost=ctx.emotionRequests.find(request=>request.target==='/api/chat');
assert.deepEqual(JSON.parse(JSON.stringify(firstEmotionPost.body)),{individual_id:'p1',question:'实际发送时采用已选情绪',context_mode:'none',emotion:'joy'});
assert.equal(firstEmotionPost.method,'POST');
ctx.finishEmotionReply({context_mode:'none',complete:true,answer:'第一条本地测试回复',reasoning:'本地测试思考'});
await emotionSend;
assert(emotionButtons.every(button=>!button.disabled));
assert.equal(run("state.chats.get('p1').emotion"),'joy','successful response leaves the selected emotion active');
assert.equal(run("state.chats.get('p1').threads.none.messages.at(-1).answer"),'第一条本地测试回复');
for(const emotion of ['none','anger']){
  emotionButton(emotion).listeners.click();run("$('chat-input').value='下一轮本地测试'");
  const pending=run('sendMessage()');await new Promise(setImmediate);
  assert.equal(ctx.emotionRequests.filter(request=>request.target==='/api/chat').at(-1).body.emotion,emotion);
  ctx.finishEmotionReply({context_mode:'none',complete:true,answer:'后续本地测试回复'});await pending;
  assert.equal(run("state.chats.get('p1').emotion"),emotion);
}
assert.equal(run("state.chats.get('p1').threads.none.messages.length"),6,'three requests append to history without clearing earlier turns');
assert.equal(run("state.chats.get('p1').threads.shared.messages[0].answer"),'必须保留的历史');
run("switchMode('shared')");assert.equal(run("state.chats.get('p1').emotion"),'anger');
run("installExtraction({individuals:[{...person,emotion:'anger'}]})");
assert.equal(run("state.chats.get('p1').emotion"),'none','a fresh workspace always starts with emotion off');
assert.equal(get('emotion-current').textContent,'当前：关闭情绪');
console.log('PASS: JS syntax; 8 synthetic samples / 56 turns; seven persistent shortcuts with busy locks; shared default/common context/isolated histories; actual incremental thought and answer fragments; in-place DOM; fold and scroll preservation; person switching and reconnect; rematch only updates actual configuration with async/person guards; receipt/result node snapshots and historical nodes retained; terminal latest-match refresh; failed partial retained but not added to context; target-only bars; measured data unchanged; text escaping; fresh-entry isolation; no-context safety; semantic disclosure; per-person emotion controls and exact chat payloads; emotion survives context/history changes and remains until closed; busy and cross-person isolation; fresh emotion reset.');

