#include"string.h"

namespace lh
{
	//string::string()
	//	:_str(new char[1] {'\0'}/*nullptr会报错，因为后面如果通过nullptr打印就是对空指针解引用*/)
	//	,_size(0)
	//	,_capacity(0)
	//{

	//}

	string::string(const char* str)
		:_size(strlen(str))
	{
		_capacity = _size;
		_str = new char[_size+1];//这里记得+1
		//strcpy(_str, str);
		memcpy(_str, str, _size + 1);
	}

	const char* string::c_str() const
	{
		return _str;
	}

	string::~string()
	{
		delete[] _str;
		_str = nullptr;
		_size = _capacity = 0;
	}

	size_t string::size() const
	{
		return _size;
	}

	char& string::operator[](size_t i)
	{
		return _str[i];
	}

	const char& string::operator[](size_t i) const
	{
		return _str[i];
	}

	string::iterator string::begin()
	{
		return _str;
	}

	string::iterator string::end()
	{
		return _str + _size;
	}

	string::const_iterator string::begin() const  //const迭代器
	{
		return _str;
	}

	string::const_iterator string::end() const
	{
		return _str + _size;
	}

	void string::reserve(size_t n)
	{
		if (n > _capacity)
		{
			char* str = new char[n + 1];
			//strcpy(str, _str);
			memcpy(str, _str, _size + 1);//记得把'\0'拷贝过去,这个函数场景更贴切string用法
			delete[] _str;
			_str = str;
			_capacity = n;//不要忘了这一步
		}
	}


	void string::push_back(char ch)
	{
		if (_size >= _capacity)
		{
			size_t newcapacity = _capacity == 0 ? 4 : 2 * _capacity;
			reserve(newcapacity);
		}

		_str[_size] = ch;
		_size++;
		_str[_size] = '\0';
	}

	void string::append(const char* str)
	{
		size_t len = strlen(str);
		if (_size + len > _capacity)//扩容逻辑
		{
			size_t newcapacity = 2 * _capacity > _size + len ? 2 * _capacity : _size + len;
			reserve(newcapacity);
		}
		//strcpy(_str + _size, str);
		memcpy(_str + _size, str, len + 1);
		_size += len;
	}

	string& string::operator+=(char ch)
	{
		push_back(ch);
		return *this;
	}

	string& string::operator+=(const char* str)
	{
		append(str);
		return *this;
	}

	ostream& operator<<(ostream& out, const string& s)
	{
		//cout << s.c_str();//这样只能按照常量字符串打印，万一追加'\0'后又有有效字符那么就打印不出来
		for (auto ch : s)//范围for
		{
			out << ch;
		}
		//也可以用for循环用下标一个一个打印
		return out;
	}

}

//上下两个都可以  可在命名空间里写好定义，也可以单独一个一个写，就是函数前面要写好命名空间
//lh::string::string()
//	:_str(new char[1] {'\0'})
//	, _size(0)
//	, _capacity(0)
//{
//}