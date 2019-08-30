# JQUERY学习笔记

## 1.使用JQUERY

首先我们要到官网去下载JQUERY的库，不同的版本提供crased与uncrased压缩与不压缩的版本，下载到本地。在本地的项目中引入JQUERY的库，编写Jquery相关的库。

未压缩的版本格式更好看，更有利于我们去阅读，而压缩过后的版本格式不好看，不利于我们去阅读。在企业的开发过程中，我们使用未压缩的版本 ，上线的时候使用压缩的版本。

## 2.JQUERY初试

首先我们要进行JQUERY的引入，与大公司保持一致，我们选择引入jquery的1.X的版本。

```html
<script src="jquery文件锁放置的位置"></script>
<script>
	//jquery的标准格式
	$(document).ready(function(){
        
	})
</script>

```

## 3.jquey与原生JS的区别

原生JS和jquery的加载模式不同，原生的JS会等到图片加载完毕之后再执行，但jquery再图片完全加载完毕之前就会执行

原生的JS如果编写多个入口函数，后面写的会覆盖前面写的

jquery后写的不会覆盖前面的

## 4.jquery入口函数的其他写法

```html
<script>
//第二种写法
jQuery(document).ready(function (){
            alert("hello world2");
        })
 //第三种写法（更推荐）
$(function(){
        
    })
 //第四种写法
jQuery(function(){
        
    })
</script>
```

## 5.jquery的冲突问题

如果使用多个框架，造成$的冲突，此时后写的会覆盖掉先写的。

我们提前声明了jQuery.noConflict()就是指我们释放了$的使用权，jquery框架中的\$都要转化成jQuery，要在所有的jQuery代码之前

自定义一个访问符号

```html
var XX = jQuery.noConflict()
```

## 6.jquery的核心函数

```html
//这就是jquery的核心函数
$()
//1.接收一个函数
alert("hello world");
//2.接受一个字符串
//2.1接受一个字符串选择器
//返回jquery对象，对象中保存了找到的dom元素
var $box1 = $(".box1")
//2.2接收一个字符串代码片段
var $p = $("<p>我是段落</p>")
//会创建对应的dom元素
//2.3接受一个dom元素
//如果我们把dom元素传给jquery，那么会把dom元素包装成为一个jquery对象，返回给我们

```

## 7.jquery对象

其实是一个伪数组，有0-length-1的属性，有length属性

## 8.静态方法和实例方法

```html
    <script>
        //声明一个类
        function AClass(){
        }
        //声明类的静态方法
        AClass.staticMethod = function(){
            alert("this is static");
        }
       //调用类的静态方法 
        AClass.staticMethod();
        //声明类的实例方法
        AClass.prototype.instanceMethod = function () {
            alert("this is  instance");
        }
        //调用类的实例方法
        //1.创建对象
        //2.通过对象调用实例方法
        var a = new AClass();
        a.instanceMethod();
    </script>
```

## 9.jquery静态each方法

```html
    <script>
        //jquery each方法遍历数组
        /*
        第一个参数：当前遍历到的索引
        第二个参数：遍历到的元素
        jquery的each方法可以用来遍历伪数组
         */
        var arr = [1,3,5,7,9];
        $.each(arr,function(index,value){
            console.log(index, value);
        })
    </script>
```

## 10.map方法

```html
    <script>
        var arr = [1,3,5,7,9];
        /*
        第一个参数是数组的名称
        第二个参数是调用的方法
        方法中第一个参数是元素
        第二个参数是遍历到的索引
        也可以遍历伪数组
         */
        $.map(arr,function(value,index){
            console.log(value, index);
        })
        /*
        each默认的返回值就是遍历谁就返回谁
        each静态方法不支持在回调函数中对数组进行处理
        map静态方法默认的返回值是一个空数组‘
        map的静态方法可以在回调函数中通过return对遍历的数组进行处理然后生成一个新的数组返回
         */
    </script>

```

## 11.jquery的其他静态方法

```html
    <script>
        var str ="     lnj      ";
        //trim用来消除字符串的两端空格
        //参数就是需要去除空格的字符串
        var res = $.trim(str);
        console.log("-----"+str);
        console.log("-----"+res);
        //判断传入的对象是否是window对象
        $.isWindow();
        //判断传入的对象是否是数组
        $.isArray();
        //判断是不是一个方法
        //jquery方法本质上是一个函数
        $.isFunction();
    </script>
```

## 12.holdready静态方法

```html
//作用是暂停ready执行
$.holdReady(true);
//在需要加载的后面写上$.holdready(false)来继续入口函数的执行
```

## 13.jquery的选择器

```html
//找到所有div中内容为空的元素
var $div1 = $('div:empty')
//从所有的div中找到有子元素或者有内容的
var $div = $('div:parent')
//从所有的div中找到内容是我是div的
var $div = $('div:contains("我是div")')
//找div中包含子元素span的div
var $div = $('div:has('span'))')
```

## 14.原生属性和属性节点

1.什么是属性
对象身上保存的变量就是属性
2.如何操作属性
对象.属性名称 = 就是给一个属性进行赋值
对象.属性名称就是获取一个属性
对象["属性名称"]也可以获取一个属性
3.什么是属性节点
\<span name="it666">\</span>
在标签中添加的属性就是属性节点,例如这里的name
4.如何操作属性节点
span.setAttribute("name","lnj")
这里的意思指的就是给找到的span元素中的name属性节点的值设置为lnj
span.getAttribute()
这里的意思是返回span元素的值
5.属性和属性节点之间有什么区别
任何对象都有属性,但是只有dom对象才有属性节点

## 15.jquery属性和属性节点

```html
    <script>
        $(function () {
            //只有一个参数的时候,是获取属性节点的功能
            $("span").attr("class");
            //如果设置两个参数,那么就是修改属性节点的值
            //注意点:无论找到多少个元素,都只会返回第一个元素的属性节点
            $("span").attr("class","box");
            //此时是修改span标签的class的值
            //无论找到多少个,都会统一地进行修改
            //如果属性节点不存在,那么此时就会在dom元素地后方添加上相应的属性节点
            //删除相对应地属性节点
            $("span").removeAttr("class");
        })
    </script>
```

