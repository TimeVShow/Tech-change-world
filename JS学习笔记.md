# JS学习笔记

HTML中的脚本必须位于\<script>之中

document.write(“”)直接向网页的\<body>中写入内容

使用var来声明变量

var cars=new Array()声明数组数组基于0

改变HTML元素的内容

document.getElementById(id).innerHTML=

改变属性document.getElementById(id).attribute

改变样式

document.getElementById(“p2”).style.property=new style

| 操作        | 效果         |
| ----------- | ------------ |
| onclick     | 单击触发     |
| onload      | 进入页面触发 |
| onunload    | 离开页面触发 |
| onmousedown | 鼠标点击时   |
| onmouseup   | 鼠标释放时   |
| onclick     | 鼠标点击时   |
|             |              |
|             |              |
|             |              |

indexOf()可以找到字符串中第一次出现某个字符的位置

JS中使用Math.round()来对数值进行四舍五入

window.open()打开新窗口

window.close()关闭当前窗口

screen.availWidth获得当前屏幕的宽度

screen.availHeight获得当前可用的屏幕的高度

window.history.back()创建后退按钮(前一章后一章的使用方法)

获得当前的时间

```javascript
function gettime()
{
	var S=new Date();
	var Y=S.getFullYear();
	var M=S.getMonth();
	var D=S.getDate();
	var h=S.getHours();
	var m=S.getMinutes();
	var s=S.getSeconds();
	h=checkTime(h);
	m=checkTime(m);
	s=checkTime(s);
	var T=Y+"年"+M+"月"+D+"日"+h+":"+m+":"+s;
	document.getElementById("time").innerHTML=T;
	t=setTimeout('gettime()',50)//一定要加，不然时间无法动态显示
}

function checkTime(i)
{
	if(i<10)
	{
		i="0"+i;
	}
	return i;
}
```







