html 学习

段落中如果需要换行则可以使用\<br />标签

在标签上style选中font-family:+字体，我们可以改变字体

font-family:字体;color:;font-size:

分别是调整字体，颜色,以及字体的大小

|    标签    |                     作用                      |
| :--------: | :-------------------------------------------: |
|    \<b>    |                     粗体                      |
|   \<big>   |                     大号                      |
|   \<em>    |                   着重文字                    |
|    \<i>    |                    斜体字                     |
|  \<small>  |                    小号字                     |
| \<strong>  |                   加重语气                    |
|   \<sub>   |                     下标                      |
|   \<sup>   |                     上标                      |
|   \<ins>   |                    下划线                     |
|   \<del>   |                    上划线                     |
|  \<html>   |              网页文件的起始标签               |
|  \<body>   |                网页主体的标签                 |
|   \<h1>    |                   网页标题                    |
|    \<p>    |                   网页段落                    |
|  \<hr />   |              生成横线，分割内容               |
|  \<br />   |                   生成空行                    |
|   \<pre>   | 保留标签内所有的空行以及缩进(常用来显示代码)  |
|  \<code>   |                常用来显示代码                 |
|   \<kbd>   |                常用来显示代码                 |
|   \<tt>    |                常用来显示代码                 |
|  \<samp>   |                常用来显示代码                 |
|   \<var>   |                 用来定义变量                  |
| \<address> | 用来定义地址,大多数网页中将该元素用斜体来表示 |
|  \<cite>   |               用来显示著作标题                |
|  \<table>  |               生成一个新的表格                |
|   \<tr>    |                表格中新的一行                 |
|   \<td>    |                 填充表格的列                  |
|   \<th>    |                定义表格的表头                 |
| \<caption> |                定义表格的标题                 |

\<abbr title="text">txt\</abbr>

<acronym title="World Wide Web"\</acronym>

两者用法相同效果也基本相同，会在页面中显示txt鼠标放置在文本上面的时候会显示出title的内容

\<bdo dir="rtl">可以改变文本显示的方向，改为从右向左

\<blockquote>浏览器可以实现自动换行

\<q>不会有特殊的呈现，会在文本两端加上双引号

文本中添加链接

\<a href="url">text\</a>

指定链接打开的方式\<a href="url"target="_blank">,在一个新的标签页打开

我们可以用\<a name="">的方式来生成一个锚,这个锚可以帮助我们快速到达网页的某个位置，可以先使用，后说明

要使用%20来替换单词之间的空格，这样我们的浏览器就可以正常的显示文本

我们使用\<img src="url">的方式在当前页面中加入一个图片，其中url指向的是文件所在的地址，而不是某个网上的地址。

\<img src="url" alt="txt">在图片加载失败的时候图片的位置最终会显示出txt的字样，这样可以方便那些使用文本浏览器的人，同时我们应该注意慎用

\<table border="1">生成一个带有表格的边框，border的值不同，生成边框的厚度也就会不相同。如果不定义，那么此时就不会显示边框

如果一个单元格为空的那么我们最好使用\<td>\&nbsp;\</td>来显示这个空白的单元格。\&nbsp为空白占位符

\<table frame="">frame中加入box,above,blow,hsides,vsides等来生成框架

\<th align="">用于表格内容排列

<table frame="box">
    <tr>
        <th align="center">love</th>
        <th align="center">like</th>
    </tr>
    <tr>
        <td align="center">h1</td>
        <td align="center">h2</td>
    </tr>
</table>



\<ul>\<li>搭配用于生成一个无序列表，单个项目前用小黑点进行标记

<ul>
    <li>Coffee</li>
    <li>Milk</li>
</ul>

有序列表使用\<ol>来进行表示

<ol>
    <li>milk</li>
    <li>coffee</li>
</ol>

如果想要实现一个带有子目录式样的列表(定义列表)我们可以使用\<dl>

\<dt>用来生成母目录，\<dd>用来生成子目录

<dl>
    <dt>Coffee</dt>
    <dd>Black hot drink</dd>
    <dt>Milk</dt>
    <dd>Whiter cold drink</dd>
</dl>

\<ul type="">可以更改项目符号的样式

\<style>\</style>之间创建一个div模板，模板的名字前要加上一个.这样才能正确地使用模板。模板的作用是为整块儿的文本内容套用模板的样式。网页的正文位于\<body>\</body>中，至于\<style>则要位于\<head>标签中。

\<span>则是一个文本的模板就是说只适用于改变行内文本的样式

正常情况下的网页通常是含有多个栏目的。我们可以使用html5中的一些他姓来帮助我们完成这个任务

| header  | 定义文档或节的页眉    |
| ------- | --------------------- |
| nav     | 定义导航链接          |
| section | 定义文档中的节        |
| article | 定义文章              |
| aside   | 定义侧边栏            |
| footer  | 定义文档或节的页脚    |
| details | 定义额外的细节        |
| summary | 定义details元素的标题 |
|         |                       |
|         |                       |

```html
<body>

<header>//可以直接引用该属性
<h1>City Gallery</h1>
</header>

<nav>
London<br>
Paris<br>
Tokyo<br>
</nav>

<section>
<h1>London</h1>
<p>
London is the capital city of England. It is the most populous city in the United Kingdom,
with a metropolitan area of over 13 million inhabitants.
</p>
<p>
Standing on the River Thames, London has been a major settlement for two millennia,
its history going back to its founding by the Romans, who named it Londinium.
</p>
</section>

<footer>
Copyright W3School.com.cn
</footer>

</body>
<style>
header {
    background-color:black;
    color:white;
    text-align:center;
    padding:5px; 
}
nav {
    line-height:30px;
    background-color:#eeeeee;
    height:300px;
    width:100px;
    float:left;//这里规定栏目的位置，左或者右
    padding:5px; 
}
section {
    width:350px;
    float:left;
    padding:10px; 
}
footer {
    background-color:black;
    color:white;
    clear:both;
    text-align:center;
    padding:5px; 
}
```

响应式网页设计:在不同的设备上显示的内容自动匹配当前设备(学习bootstrap)

Iframe我们可以使用iframe来在当前的页面上显示另一个页面的内容。

使用方法\<iframe src="url" width="" height="">width,height用来定义高度以及宽度

我们可以通过设置frameborder="0"的方式来去掉iframe的边框

```html
<iframe src="http://www.allmight.xyz" name="test" frameborder="0"></iframe>
		<p>
			<a href="http://www.allmight.xyz" target="test">test</a?
		</p>
```

通过上面的操作我们最终可以将指向的链接在iframe框架中打开

以下元素定义在\<head>中

\<title>规定网页的标题,也就是在浏览器标签栏中显示的内容

\<base>可以为文章中所有链接打开的位置指定一个默认目标

\<meta>元素是关于数据的信息

```html
<!--来定义作者和日期-->
<meta name="author" content="">
<meta name="revised" content="">

```

还可以用来浏览器

```html
<!--如何重定向当前页面-->
<!DOCTYPE HTML>
<html>
	<head>
		<meta http-equiv="Content-Type" content="text/html; 		charset=gb2312" />
		<meta http-equiv="Refresh" 				  	  content="5;url=http://www.w3school.com.cn" />
	</head>

<body>
	<p>
		对不起。我们已经搬家了。您的 URL 是 <a	 href="http://www.w3school.com.cn">http://www.w3school.com.cn	</a>
	</p>

	<p>您将在 5 秒内被重定向到新的地址。</p>

	<p>如果超过 5 秒后您仍然看到本消息，请点击上面的链接。</p>

	</body>
</html>
```

*\<input type="submit">* 定义用于向*表单处理程序*（form-handler）*提交*表单的按钮。

表单处理程序通常是包含用来处理输入数据的脚本的服务器页面。

\<form action="">这里定义在提交表单时执行的动作

method为表单提交确定方法,通常只有两种,post&&get,get的安全度较低,post的安全度更高

如果想要正确地提交那么我们需要保证input中必须包含name元素





