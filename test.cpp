#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<vector>
#include<list>
#include<string>
#include<algorithm>
using namespace std;

//template<typename T>
//class stack {
//public:
//	stack(int n = 4)
//		:_a(new T[n])
//		, _top(0)
//		, _capacity(n)
//	{
//		cout << "stack(int n = 4)" << endl;
//	}
//
//	void push(const T& x);
//
//	~stack()
//	{
//		delete[] _a;
//		_a = nullptr;
//		_top = 0;
//		_capacity = 0;
//
//		cout << "~stack()" << endl;
//	}
//private:
//	T* _a;
//	int _top = 0;
//	int _capacity = 4;
//};
//
//template<typename K>
//void stack<K>::push(const K& x)//K和T不重要，因为这个相当于形参
//{
//	_a[_top++] = x;
//}
//
//int main()
//{
//	stack<int> st1;//显式实例化后才是类型
//	stack<double> st2;
//	
//	st1.push(1);
//	st1.push(2);
//	st1.push(3);
//	st1.push(4);
//
//	st2.push(1.1);
//	st2.push(2.2);
//
//
//	return 0;
//}

//-------------------------------------------------------------------------------
//string是类模板实例出来的  typedef basic_string<char> string; 所以是管理字符数组的顺序表  utf-8  basic_string(是类模板)
//string历史比STL早，所以不在容器里，但是纯分类来讲可以是容器（也算一种数据结构）
//STL有六大件，重要的是容器和算法

//class stack
//{
//private:
//	char* _a;
//	int top;
//	int capacity;
//};


void string_test1()
{
	string s1;//默认构造  //重点
	string s2("Hello world!");//传常量字符串构造  //重点
	string s3(s2);//拷贝构造  //重点
	string s4(s2, 2, 6);//拷贝构造一部分，第一个数字是指起始位置，第二个字是拷贝构造长度  
	string s5(s2, 1, 60);
	string s6(s2, 1);

	const char* p = "Hello";
	string s7(p, 3);//复制字符串前 n 个字符
	string s8(100, '#');//复制字符 100 次  

	s1 = s2;//赋值重载
	s1 = "HELLO";//赋值重载
	s1 = 's';//赋值重载

	cout << s1 << endl;//流插入提取符，已经运算符重载，所以能够支持string这个自定义类型
	cout << s2 << endl;
	cout << s3 << endl;
	cout << s4 << endl;
	cout << s5 << endl;
	cout << s6 << endl;
	cout << s7 << endl;
	cout << s8 << endl;

	int a[5] = {0};
}

void string_test2()
{
	
	
	string s1("hello world");
	//s1[0]++;
	cout << s1 << endl;

	//遍历+修改
	//1.operator[]运算符重载，（小众，只有string，vector底层是数组的 才支持 []重载 才可以用这种方式遍历）
	for (size_t i = 0; i < s1.size(); i++)
	{
		s1[i]++;
	}
	cout << s1 << endl;


	//2.迭代器（所有容器的主流遍历方式+修改）   想象成像指针一样的东西  iterator是类型
	string::iterator it1 = s1.begin();//begin是起始位置
	while (it1 != s1.end())//end是最后一个数据的下一个位置（串的最后字符的下一个位置也就是'\0'）  故[begain,end)才是完整的串
	{
		(*it1)--;
		it1++;
	}

	cout << s1 << endl;


	//vector是顺序表 类模板 要显式实例化
	vector<int> v;
	v.push_back(1);
	v.push_back(2);
	v.push_back(3);
	vector<int>::iterator it2 = v.begin();
	while (it2 != v.end())
	{
		cout << *it2 << " " ;
		it2++;
	}
	cout << endl;


	//list是链表
	list<int> lt;
	lt.push_back(1);
	lt.push_back(2);
	lt.push_back(3);
	list<int>::iterator it3 = lt.begin();
	while (it3 != lt.end())
	{
		cout << *it3 << " ";
		it3++;
	}
	cout << endl;

	//迭代器的意义
	//1.用统一类似的方式遍历修改容器
	//2.可以用以迭代器的方式传给算法  （算法脱离具体的底层结构，和底层结构解耦）
	//算法 函数模板实现针对多个容器处理
	reverse(s1.begin(), s1.end());//就是函数模板
	reverse(v.begin(), v.end());
	reverse(lt.begin(), lt.end());

	cout << s1 << endl;

	// 3、范围for C++11  语法糖
	auto x = 20;
	auto y = 10.5;
	cout << x << endl;
	cout << y << endl;

	int& z = x;//z是x的别名
	auto m = z;//这种只是把z的值赋给了m，故m还是int类型
	m++;
	cout << "x = " << x << endl;
	cout << "z = " << z << endl;
	cout << "m = " << m << endl;

	auto& n = z;//这种n就是int&,n就是x的别名
	n++;
	cout << "x = " << x << endl;
	cout << "z = " << z << endl;
	cout << "n = " << n << endl;


	// 自动取容器数据赋值给e
	// 自动迭代
	// 自动判断结束
	//for (auto e : s1)
	//{
	//	cout << e << ' ';
	//}
	//cout << endl;

	//for (auto e : s1)//与m一样
	//{
	//	e--;
	//}
	
	//for (char& e : s1)//也可以这样，但是一般是下面的情况
	for (auto& e : s1)//与n一样
	{
		e--;
	}
	for (auto e : s1)
	{
		cout << e << ' ';
	}
	cout << endl;
	cout << s1 << endl;

	for (auto e : v)
	{
		cout << e << ' ';
	}
	cout << endl;

	for (auto e : lt)
	{
		cout << e << ' ';
	}
	cout << endl;

}

void string_test3()
{
	string s1("123456");
	const string s2("hello world");

	// [begin(), end())
	// 2、迭代器 （所有容器主流遍历+修改方式）
	//  s1的的每个字符都--
	string::iterator it1 = s1.begin();
	while (it1 != s1.end())
	{
		(*it1)--;
		++it1;
	}
	cout << s1 << endl;

	string::const_iterator it2 = s2.begin();//const
	while (it2 != s2.end())
	{
		// (*it2)--; 不能修改
		++it2;
	}
	cout << s2 << endl;

	string::reverse_iterator it3 = s1.rbegin();
	while (it3 != s1.rend())
	{
		cout << *it3 << " ";
		++it3;
	}
	cout << endl;

	string::const_reverse_iterator it4 = s2.rbegin();
	while (it4 != s2.rend())
	{
		// 不能修改
		// *it4 = 'x';

		cout << *it4 << " ";
		++it4;
	}
	cout << endl;

	for (auto ch : s2)
	{
		cout << ch << " ";
	}
	cout << endl;
}

void string_test4()
{
	string s1("123456");

	//均不包含'\0'
	cout << s1.size() << endl;
	cout << s1.capacity() << endl;
	cout << s1.length() << endl;
	
	string s2;
	//// 确定知道需要多少空间，提前开好，避免扩容，提高效率
	//s2.reserve(100);//开空间的意思  //预订；预留
	size_t old = s2.capacity();
	cout << "capacity:" << old << endl;
	for (size_t i = 0; i < 100; i++)
	{
		s2.push_back('x');//除了第一次是2倍，其余是1.5倍
		if (s2.capacity() != old)
		{
			old = s2.capacity();
			cout << "capacity:>" << s2.capacity() << endl;
		}
	}

	cout << s1 << endl;
	s1.clear();
	cout << s1 << endl;

	cout <<"s2.size() :>" << s2.size() << endl;
	cout <<"s2.capacity() :>" << s2.capacity() << endl;
	//s2.clear();//与下面效果相同
	for (size_t i = 0; i < 100; i++)
	{
		s2.pop_back();
	}
	s2.shrink_to_fit();//用于size变小，但是capacity没有变小的情况，所以出现了缩容量的函数，但是这个函数不推荐使用，因为它是时间换空间。重新开空间，拷贝和释放原来的空间
	cout <<"s2.size() :>" << s2.size() << endl;
	cout <<"s2.capacity() :>" << s2.capacity() << endl;
}

void string_test5()
{
	string s1("123456");
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// 扩容  有关capacity
	s1.reserve(100);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// 缩容 不靠谱
	s1.reserve(3);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// 缩容 不靠谱
	s1.reserve(10);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl << endl;



	string s2("Hello world!");
	cout << "string:>"<<s2 << endl;
	cout << "size:" << s2.size() << endl;//12
	cout << "capacity:" << s2.capacity() << endl;//15
	
	// 插入数据，让size到n个
	// n > capacity > size;
	s2.resize(20,'x');//直接影响size，size影响capacity
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;

	// capacity > n > size;
	//s2.resize(25, 'x');//可传可不传，传了就用这个填充
	s2.resize(25);
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;

	// capacity > size > n;//删除数据
	s2.resize(5);
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;
}

void string_test6()
{
	string s1("Hello");

	s1[0]--;//operator[]运算符重载，越界是通过assert判断
	cout << s1 << endl;

	s1.at(0)++;//这个函数对于越界的操作是做出抛异常的行为
	cout << s1 << endl;


	char& s2 = s1.back();
	//s1.back() = '*';
	s2 = '*';
	cout << s1 << endl;
}

void string_test7()
{
	string s1("Hello");
	s1 += ' ';
	s1 += "world";
	cout << s1 << endl;

	s1.append("!");
	cout << s1 << endl;
	string s2 = " wonderful";
	//s1.append(s2);
	//cout << s1 << endl;
	s1.append(s2,3,4);//后面的数字是长度len
	cout << s1 << endl;
	s1.append(" city", 3);
	cout << s1 << endl;

	s1 = 'x';
	cout << s1 << endl;
	s1.assign(s2);
	cout << s1 << endl;
	s1.assign(10, 'y');
	cout << s1 << endl;

	// insert 谨慎使用，底层数据挪动，效率低下，O(N)
	string s3("hello world");
	s3.insert(0, "yyy");
	cout << s3 << endl;

	s3.insert(0,"!");
	cout << s3 << endl;

	s3.insert(0, 2,'!');
	cout << s3 << endl;

	s3.insert(s3.begin(), '!');
	cout << s3 << endl;

}

void string_test8()
{
	// erase 谨慎使用，底层数据挪动，效率低下，O(N)
	string s3("hello world");
	s3.erase(5, 1);
	cout << s3 << endl;

	s3.erase(5);
	cout << s3 << endl;

	// replace 谨慎使用，底层数据挪动，效率低下，O(N)
	string s4("hello   world");
	cout << s4 << endl;
	s4.replace(5, 3, "#");//下标5的位置，将长度3的字符串替换为"#"
	cout << s4 << endl;

	s4.replace(5, 1, "!!!!!!!!!");
	cout << s4 << endl;

	//空格换成两个%%

	string s1("hello  world hello world hello world lh");
	size_t pos = s1.find(' ');
	while (pos != string::npos)
	{
		s1.replace(pos,1, "%%");
		pos = s1.find(' ',pos+2);
	}
	cout << s1 << endl;

	string s2(" hello world lh");
	string s;
	s.reserve(s2.size());
	for (auto ch : s2)
	{
		if (ch == ' ')
			s += "%%";
		else
			s += ch;
	}
	cout << s << endl;
}

//自己优化后的  //但是其实你倒着找不就是更简单吗，然后库里面还有rfind
//string subsuffix(const string& filename)
//{
//	size_t pos = filename.find('.');
//	size_t poslast = string::npos;
//	while (pos != string::npos)
//	{
//		poslast = pos;
//		pos = filename.find('.',pos+1);
//	}
//	if (poslast != string::npos)
//	{
//		return filename.substr(poslast);
//	}
//	else
//	{
//		//return string();
//		return "";//隐式类型转换
//	}
//}

string subsuffix(const string& filename)
{
	size_t pos = filename.rfind('.');

	if (pos != string::npos)
	{
		return filename.substr(pos);
	}
	else
	{
		//return string();
		return "";//隐式类型转换
	}
}



void string_test9()
{
	//c_str copy substr
	//string filename("test.cpp");
	//FILE*str=fopen(filename.c_str(), "r");
	//if (str == nullptr)
	//{
	//	cout << "fopen error" << endl;
	//	exit(1);
	//}
	//char ch = fgetc(str);
	//while (ch != EOF)
	//{
	//	cout << ch;
	//	ch = fgetc(str);
	//}

	//-----------------------------------------

	//char buffer[20];
	//std::string str("Test string...");
	//std::size_t length = str.copy(buffer, 6, 5);//不常用，且返回值是长度
	//buffer[length] = '\0';//要补\0
	//std::cout << "buffer contains: " << buffer << '\n';

	//-----------------------------------------

	string filename1("test.cpp");
	//string s1 = filename1.substr(4, filename1.size() - 4);
	/*string s1 = filename1.substr(4);*/
	string s1 = subsuffix(filename1);
	cout << s1<<endl;

	string filename2("test");
	string filename3("test.cpp.....txt.c");
	string s2 = subsuffix(filename2);
	string s3 = subsuffix(filename3);
	cout << s2 << endl;
	cout << s3 << endl;

}

//void split_url(const string& url)
//{
//	size_t pos = url.find(':');
//	if (pos != string::npos)
//	{
//		cout << url.substr(0, pos) << endl;
//	}
//	size_t pos2 = pos + 3;
//	size_t pos3 = url.find('/', pos2);
//	if (pos3 != string::npos)
//	{
//		cout << url.substr(pos2, pos3 - pos2) << endl;
//		cout << url.substr(pos3) << endl << endl;
//	}
//
//}

vector<string> split_url(const string& url)
{
	vector<string> v;
	size_t pos = url.find(':');
	if (pos != string::npos)
	{
		v.push_back(url.substr(0, pos));
	}
	size_t pos2 = pos + 3;
	size_t pos3 = url.find('/', pos2);
	if (pos3 != string::npos)
	{
		v.push_back(url.substr(pos2, pos3 - pos2));
		v.push_back(url.substr(pos3));
	}
	return v;
}
//void string_test()
//{
//	string url1 = "http://legacy.cplusplus.com/reference/string/string/";
//	string url2 = "https://yuanbao.tencent.com/chat/naQivTmsDa/43735652-b5e3-11ef-bcaa-c6162ee89a56?yb_channel=3003";
//	string url3 = "https://legacy.cplusplus.com/reference/vector/vector/";
//
//	//split_url(url1);
//	//split_url(url2);
//	//split_url(url3);
//	vector<string> ret = split_url(url2);
//	cout << ret[2] << endl;
//	// string ip = dns_request(ret[1].c_str());
//}

void string_test10()
{
	std::string str("There are two needles in this haystack with needles.");
	std::string str2("needle");

	// different member versions of find in the same order as above:
	std::size_t found = str.find(str2);
	if (found != std::string::npos)
		std::cout << "first 'needle' found at: " << found << '\n';

	found = str.find("needles are small", found +1, 6);//found+1是字符串查找起始位置，也就是上一次的下一个字符
	if (found != std::string::npos)
		std::cout << "second 'needle' found at: " << found << '\n';
}

void string_test11()
{
	//std::string str("Please, replace the vowels in this sentence by asterisks.");
	//std::size_t found = str.find_first_of("aeiou");
	////std::size_t found = str.find_first_not_of("aeiou");
	//while (found != std::string::npos)
	//{
	//	str[found] = '*';
	//	found = str.find_first_of("aeiou", found + 1);
	//	//found = str.find_first_not_of("aeiou", found + 1);
	//}

	//std::cout << str << '\n';

	string s1("xxx");
	string s2("xxxxxxy");
	const char* str="22223333";
	//s1 == s2;
	//s1 == str;
	//str == s2;

	//s1 + s2;
	//s1 + str;
	//str + s1;

	cin >> s1;
	cout << s1;

}

//int main()
//{
//	//string_test4();
//	//string_test5();
//	//string_test7();
//	//string_test8();
//	//string_test9();
//	string_test11();
//
//	return 0;
//}

//int main()
//{
//	string s1;
//	//cin >> s1;有空格就会读取断掉 
//	getline(cin, s1);//这个是直接读取一行，换行才停止读取
//	size_t pos = s1.rfind(' ');
//	if (pos != string::npos)
//	{
//		cout << s1.size() - pos - 1;
//	}
//	else {
//		cout << s1.size();
//	}
//	return 0;
//}


//int main()
//{
//	string s1;
//	//ctrl + Z + enter结束循环
//	while (cin >> s1)
//	{
//		;
//	}
//	cout << s1<<endl;
//	return 0;
//}

#include"string.h"
namespace lh//不放到命名空间里的话，就无法测试，因为出去命名空间，函数就会优先去全局找，那么我们自己写的就不行了
{
	void string_test1()
	{
		string s1; 
		cout << s1.c_str() << endl;
		string s2("hello world");
		cout<< s2.c_str()<<endl;//返回地址后cout打印出来全文

		for (size_t i = 0; i < s2.size(); i++)
		{
			s2[i]++;
		}
		cout << s2.c_str() << endl;

		for (auto& ch : s2)//范围for需要实现begin end函数就行
		{
			ch--;
			cout << ch << ' ';
		}
		cout << endl;

		string::iterator it = s2.begin();//迭代器
		while (it != s2.end())
		{
			(*it)++;
			cout << *it << ' ';
			it++;
		}

		cout << endl;
		const string s3("hello world");
		string::const_iterator it3 = s3.begin();//迭代器
		while (it3 != s3.end())
		{
			cout << *it3 << ' ';
			it3++;
		}
	}

	void string_test2()
	{
		string s("Hello world!");
		s.push_back('x');
		cout << s.c_str() << endl;

		s.append("lh太聪明了！");
		cout << s.c_str() << endl;

		s += 'c';
		s += "c太聪明了！";
		cout << s.c_str() << endl<<endl;

		string s1("Hello world");//可以重新弄std命名空间域用来验证我们写的代码
		s1 += '\0';
		s1 += '\0';
		s1 += '!';
		cout << s1 << endl;

		s1 += "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy";//按照写的代码有乱码,strcpy函数应对不了后面追加了'\0'然后又追加有效字符，无法准确copy
		cout << s1 << endl;
		cout << s1.c_str() << endl;

	}
}

int main()
{
	lh::string_test2();

	return 0;
}