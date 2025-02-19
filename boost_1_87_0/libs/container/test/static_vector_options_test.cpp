//////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright Ion Gaztanaga 2004-2013. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/container for documentation.
//
//////////////////////////////////////////////////////////////////////////////
#define BOOST_ENABLE_ASSERT_HANDLER
#include <boost/container/static_vector.hpp>
#include <boost/core/lightweight_test.hpp>
#include <new> //for bad_alloc
#include <boost/assert.hpp>
#include <cstdlib>
using namespace boost::container;

//User-defined assertion to test throw_on_overflow
struct throw_on_overflow_off
{};

namespace boost {
   void assertion_failed(char const *, char const *, char const *, long)
   {
      #ifdef BOOST_NO_EXCEPTIONS
      std::abort();
      #else
      throw throw_on_overflow_off();
      #endif
   }

   void assertion_failed_msg(char const *, char const *, char const *, char const *, long )
   {
      #ifdef BOOST_NO_EXCEPTIONS
      std::abort();
      #else
      throw throw_on_overflow_off();
      #endif
   }
}

void test_alignment()
{
   const std::size_t Capacity = 10u;
   {  //extended alignment
      const std::size_t extended_alignment = sizeof(int)*4u;
      BOOST_CONTAINER_STATIC_ASSERT(extended_alignment > dtl::alignment_of<int>::value);
      #if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
      using options_t = static_vector_options_t< inplace_alignment<extended_alignment> >;
      #else
      typedef static_vector_options
         < inplace_alignment<extended_alignment> >::type options_t;
      #endif

      static_vector<int, Capacity, options_t> v;
      v.resize(v.capacity());
      BOOST_ASSERT((reinterpret_cast<std::size_t>(&v[0]) % extended_alignment) == 0);
   }
   {  //default alignment
      #if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
      using options_t = static_vector_options_t< inplace_alignment<0> >;
      #else
      typedef static_vector_options< inplace_alignment<0> >::type options_t;
      #endif

      static_vector<int, Capacity, options_t> v;
      v.resize(v.capacity());
      BOOST_ASSERT((reinterpret_cast<std::size_t>(&v[0]) % dtl::alignment_of<int>::value) == 0);
   }
}

void test_throw_on_overflow()
{
   #if !defined(BOOST_NO_EXCEPTIONS)
   const std::size_t Capacity = 10u;
   {  //throw_on_overflow == true, expect bad_alloc
      #if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
      using options_t = static_vector_options_t< throw_on_overflow<true> >;
      #else
      typedef static_vector_options
         < throw_on_overflow<true> >::type options_t;
      #endif

      static_vector<int, Capacity, options_t> v;

      v.resize(Capacity);
      bool expected_type_thrown = false;

      BOOST_CONTAINER_TRY{
         v.push_back(0);
      }
      BOOST_CONTAINER_CATCH(bad_alloc_t&)
      {
         expected_type_thrown = true;
      }
      BOOST_CONTAINER_CATCH(...)
      {}
      BOOST_CONTAINER_CATCH_END

      BOOST_TEST(expected_type_thrown == true);
      BOOST_TEST(v.capacity() == Capacity);
   }
   {  //throw_on_overflow == false, test it through BOOST_ASSERT
      //even in release mode (BOOST_ENABLE_ASSERT_HANDLER), and throwing
      //a special type in that assertion.
      #if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
      using options_t = static_vector_options_t< throw_on_overflow<false> >;
      #else
      typedef static_vector_options< throw_on_overflow<false> >::type options_t;
      #endif

      static_vector<int, Capacity, options_t> v;

      v.resize(Capacity);
      bool expected_type_thrown = false;

      BOOST_CONTAINER_TRY{
         v.push_back(0);
      }
      BOOST_CONTAINER_CATCH(throw_on_overflow_off)
      {
         expected_type_thrown = true;
      }
      BOOST_CONTAINER_CATCH(...)
      {}
      BOOST_CONTAINER_CATCH_END

      BOOST_TEST(expected_type_thrown == true);
      BOOST_TEST(v.capacity() == Capacity);
   }
   #endif
}

template<class Unsigned, class VectorType>
void test_stored_size_type_impl()
{
   #ifndef BOOST_NO_EXCEPTIONS
   VectorType v;
   typedef typename VectorType::size_type    size_type;
   typedef typename VectorType::value_type   value_type;
   size_type const max = Unsigned(-1);
   v.resize(5);
   v.resize(max);
   BOOST_TEST_THROWS(v.resize(max+1),                    std::exception);
   BOOST_TEST_THROWS(v.push_back(value_type(1)),         std::exception);
   BOOST_TEST_THROWS(v.insert(v.begin(), value_type(1)), std::exception);
   BOOST_TEST_THROWS(v.emplace(v.begin(), value_type(1)),std::exception);
   BOOST_TEST_THROWS(v.reserve(max+1),                   std::exception);
   BOOST_TEST_THROWS(VectorType v2(max+1),               std::exception);
   #endif
}

template<class Unsigned>
void test_stored_size_type()
{
   #if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
   using options_t = static_vector_options_t< stored_size<Unsigned> >;
   #else
   typedef typename static_vector_options
      < stored_size<Unsigned> >::type options_t;
   #endif

   typedef static_vector<unsigned char, Unsigned(-1)> normal_static_vector_t;

   {
      typedef static_vector<unsigned char, Unsigned(-1), options_t> static_vector_t;
      BOOST_CONTAINER_STATIC_ASSERT(sizeof(normal_static_vector_t) > sizeof(static_vector_t));
      test_stored_size_type_impl<Unsigned, static_vector_t>();
   }
}

int main()
{
   test_alignment();
   test_throw_on_overflow();
   test_stored_size_type<unsigned char>();
   test_stored_size_type<unsigned short>();
   return ::boost::report_errors();
}
