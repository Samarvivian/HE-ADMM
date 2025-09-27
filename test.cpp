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
//void stack<K>::push(const K& x)//K��T����Ҫ����Ϊ����൱���β�
//{
//	_a[_top++] = x;
//}
//
//int main()
//{
//	stack<int> st1;//��ʽʵ�������������
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
//string����ģ��ʵ��������  typedef basic_string<char> string; �����ǹ����ַ������˳���  utf-8  basic_string(����ģ��)
//string��ʷ��STL�磬���Բ�����������Ǵ���������������������Ҳ��һ�����ݽṹ��
//STL�����������Ҫ�����������㷨

//class stack
//{
//private:
//	char* _a;
//	int top;
//	int capacity;
//};


void string_test1()
{
	string s1;//Ĭ�Ϲ���  //�ص�
	string s2("Hello world!");//�������ַ�������  //�ص�
	string s3(s2);//��������  //�ص�
	string s4(s2, 2, 6);//��������һ���֣���һ��������ָ��ʼλ�ã��ڶ������ǿ������쳤��  
	string s5(s2, 1, 60);
	string s6(s2, 1);

	const char* p = "Hello";
	string s7(p, 3);//�����ַ���ǰ n ���ַ�
	string s8(100, '#');//�����ַ� 100 ��  

	s1 = s2;//��ֵ����
	s1 = "HELLO";//��ֵ����
	s1 = 's';//��ֵ����

	cout << s1 << endl;//��������ȡ�����Ѿ���������أ������ܹ�֧��string����Զ�������
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

	//����+�޸�
	//1.operator[]��������أ���С�ڣ�ֻ��string��vector�ײ�������� ��֧�� []���� �ſ��������ַ�ʽ������
	for (size_t i = 0; i < s1.size(); i++)
	{
		s1[i]++;
	}
	cout << s1 << endl;


	//2.����������������������������ʽ+�޸ģ�   �������ָ��һ���Ķ���  iterator������
	string::iterator it1 = s1.begin();//begin����ʼλ��
	while (it1 != s1.end())//end�����һ�����ݵ���һ��λ�ã���������ַ�����һ��λ��Ҳ����'\0'��  ��[begain,end)���������Ĵ�
	{
		(*it1)--;
		it1++;
	}

	cout << s1 << endl;


	//vector��˳��� ��ģ�� Ҫ��ʽʵ����
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


	//list������
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

	//������������
	//1.��ͳһ���Ƶķ�ʽ�����޸�����
	//2.�������Ե������ķ�ʽ�����㷨  ���㷨�������ĵײ�ṹ���͵ײ�ṹ���
	//�㷨 ����ģ��ʵ����Զ����������
	reverse(s1.begin(), s1.end());//���Ǻ���ģ��
	reverse(v.begin(), v.end());
	reverse(lt.begin(), lt.end());

	cout << s1 << endl;

	// 3����Χfor C++11  �﷨��
	auto x = 20;
	auto y = 10.5;
	cout << x << endl;
	cout << y << endl;

	int& z = x;//z��x�ı���
	auto m = z;//����ֻ�ǰ�z��ֵ������m����m����int����
	m++;
	cout << "x = " << x << endl;
	cout << "z = " << z << endl;
	cout << "m = " << m << endl;

	auto& n = z;//����n����int&,n����x�ı���
	n++;
	cout << "x = " << x << endl;
	cout << "z = " << z << endl;
	cout << "n = " << n << endl;


	// �Զ�ȡ�������ݸ�ֵ��e
	// �Զ�����
	// �Զ��жϽ���
	//for (auto e : s1)
	//{
	//	cout << e << ' ';
	//}
	//cout << endl;

	//for (auto e : s1)//��mһ��
	//{
	//	e--;
	//}
	
	//for (char& e : s1)//Ҳ��������������һ������������
	for (auto& e : s1)//��nһ��
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
	// 2�������� ������������������+�޸ķ�ʽ��
	//  s1�ĵ�ÿ���ַ���--
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
		// (*it2)--; �����޸�
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
		// �����޸�
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

	//��������'\0'
	cout << s1.size() << endl;
	cout << s1.capacity() << endl;
	cout << s1.length() << endl;
	
	string s2;
	//// ȷ��֪����Ҫ���ٿռ䣬��ǰ���ã��������ݣ����Ч��
	//s2.reserve(100);//���ռ����˼  //Ԥ����Ԥ��
	size_t old = s2.capacity();
	cout << "capacity:" << old << endl;
	for (size_t i = 0; i < 100; i++)
	{
		s2.push_back('x');//���˵�һ����2����������1.5��
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
	//s2.clear();//������Ч����ͬ
	for (size_t i = 0; i < 100; i++)
	{
		s2.pop_back();
	}
	s2.shrink_to_fit();//����size��С������capacityû�б�С����������Գ������������ĺ�������������������Ƽ�ʹ�ã���Ϊ����ʱ�任�ռ䡣���¿��ռ䣬�������ͷ�ԭ���Ŀռ�
	cout <<"s2.size() :>" << s2.size() << endl;
	cout <<"s2.capacity() :>" << s2.capacity() << endl;
}

void string_test5()
{
	string s1("123456");
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// ����  �й�capacity
	s1.reserve(100);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// ���� ������
	s1.reserve(3);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl;

	// ���� ������
	s1.reserve(10);
	cout << "size:" << s1.size() << endl;
	cout << "capacity:" << s1.capacity() << endl << endl;



	string s2("Hello world!");
	cout << "string:>"<<s2 << endl;
	cout << "size:" << s2.size() << endl;//12
	cout << "capacity:" << s2.capacity() << endl;//15
	
	// �������ݣ���size��n��
	// n > capacity > size;
	s2.resize(20,'x');//ֱ��Ӱ��size��sizeӰ��capacity
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;

	// capacity > n > size;
	//s2.resize(25, 'x');//�ɴ��ɲ��������˾���������
	s2.resize(25);
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;

	// capacity > size > n;//ɾ������
	s2.resize(5);
	cout << "string:>" << s2 << endl;
	cout << "size:" << s2.size() << endl;
	cout << "capacity:" << s2.capacity() << endl;
}

void string_test6()
{
	string s1("Hello");

	s1[0]--;//operator[]��������أ�Խ����ͨ��assert�ж�
	cout << s1 << endl;

	s1.at(0)++;//�����������Խ��Ĳ������������쳣����Ϊ
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
	s1.append(s2,3,4);//����������ǳ���len
	cout << s1 << endl;
	s1.append(" city", 3);
	cout << s1 << endl;

	s1 = 'x';
	cout << s1 << endl;
	s1.assign(s2);
	cout << s1 << endl;
	s1.assign(10, 'y');
	cout << s1 << endl;

	// insert ����ʹ�ã��ײ�����Ų����Ч�ʵ��£�O(N)
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
	// erase ����ʹ�ã��ײ�����Ų����Ч�ʵ��£�O(N)
	string s3("hello world");
	s3.erase(5, 1);
	cout << s3 << endl;

	s3.erase(5);
	cout << s3 << endl;

	// replace ����ʹ�ã��ײ�����Ų����Ч�ʵ��£�O(N)
	string s4("hello   world");
	cout << s4 << endl;
	s4.replace(5, 3, "#");//�±�5��λ�ã�������3���ַ����滻Ϊ"#"
	cout << s4 << endl;

	s4.replace(5, 1, "!!!!!!!!!");
	cout << s4 << endl;

	//�ո񻻳�����%%

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

//�Լ��Ż����  //������ʵ�㵹���Ҳ����Ǹ�����Ȼ������滹��rfind
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
//		return "";//��ʽ����ת��
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
		return "";//��ʽ����ת��
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
	//std::size_t length = str.copy(buffer, 6, 5);//�����ã��ҷ���ֵ�ǳ���
	//buffer[length] = '\0';//Ҫ��\0
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

	found = str.find("needles are small", found +1, 6);//found+1���ַ���������ʼλ�ã�Ҳ������һ�ε���һ���ַ�
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
//	//cin >> s1;�пո�ͻ��ȡ�ϵ� 
//	getline(cin, s1);//�����ֱ�Ӷ�ȡһ�У����в�ֹͣ��ȡ
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
//	//ctrl + Z + enter����ѭ��
//	while (cin >> s1)
//	{
//		;
//	}
//	cout << s1<<endl;
//	return 0;
//}

#include"string.h"
namespace lh//���ŵ������ռ���Ļ������޷����ԣ���Ϊ��ȥ�����ռ䣬�����ͻ�����ȥȫ���ң���ô�����Լ�д�ľͲ�����
{
	void string_test1()
	{
		string s1; 
		cout << s1.c_str() << endl;
		string s2("hello world");
		cout<< s2.c_str()<<endl;//���ص�ַ��cout��ӡ����ȫ��

		for (size_t i = 0; i < s2.size(); i++)
		{
			s2[i]++;
		}
		cout << s2.c_str() << endl;

		for (auto& ch : s2)//��Χfor��Ҫʵ��begin end��������
		{
			ch--;
			cout << ch << ' ';
		}
		cout << endl;

		string::iterator it = s2.begin();//������
		while (it != s2.end())
		{
			(*it)++;
			cout << *it << ' ';
			it++;
		}

		cout << endl;
		const string s3("hello world");
		string::const_iterator it3 = s3.begin();//������
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

		s.append("lh̫�����ˣ�");
		cout << s.c_str() << endl;

		s += 'c';
		s += "c̫�����ˣ�";
		cout << s.c_str() << endl<<endl;

		string s1("Hello world");//��������Ūstd�����ռ���������֤����д�Ĵ���
		s1 += '\0';
		s1 += '\0';
		s1 += '!';
		cout << s1 << endl;

		s1 += "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy";//����д�Ĵ���������,strcpy����Ӧ�Բ��˺���׷����'\0'Ȼ����׷����Ч�ַ����޷�׼ȷcopy
		cout << s1 << endl;
		cout << s1.c_str() << endl;

	}
}

int main()
{
	lh::string_test2();

	return 0;
}