#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string.h>
using namespace std;
namespace lh
{
	class string
	{
	public:
		typedef char* iterator;
		typedef const char* const_iterator;
		iterator begin();
		iterator end();

		const_iterator begin() const;
		const_iterator end() const;
		//string();

		string(const char*str="");//这个直接包含进上一个函数了
		const char* c_str() const;

		~string();

		size_t size() const;
		char& operator[](size_t i);
		const char& operator[](size_t i) const;


		void reserve(size_t n);

		void push_back(char ch);
		void append(const char* str);

		string& operator+=(char ch);
		string& operator+=(const char* str);


	private:
		char* _str;
		size_t _size;//有效大小 不包括'\0'
		size_t _capacity;//有效容量'\0'
	};

	ostream& operator<<(ostream& out, const string& s);
}