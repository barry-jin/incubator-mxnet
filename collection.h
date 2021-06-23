/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2016 by Contributors
* \file collection.h
* \brief definition of some python built-in containers
* \author Zhenghui Jin
*/

#ifndef MXNET_CPP_COLLECTION_H_
#define MXNET_CPP_COLLECTION_H_

#include <string>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include "mxnet-cpp/base.h"

namespace mxnet {
namespace cpp {

/*!
 * \brief OrderedDict container .
 * \tparam K The key type.
 * \tparam V The value type.
 */
template <typename K, typename V>
class OrderedDict {
 public:
  class iterator;
  /*!
   * \brief default constructor
   */
  OrderedDict();
  /*!
   * \brief move constructor
   * \param other source
   */
  OrderedDict(OrderedDict<K, V>&& other) = default;
  /*!
   * \brief copy constructor
   * \param other source
   */
  OrderedDict(const OrderedDict<K, V>& other)
      : _keys(other._keys), _values(other._values) {}
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  OrderedDict<K, V>& operator=(OrderedDict<K, V>&& other) = default;
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  OrderedDict<K, V>& operator=(const OrderedDict<K, V>& other) {
    _keys = other._keys;
    _values = other._values;
    return *this;
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  OrderedDict(std::initializer_list<std::pair<K, V>> init) {
    _values.reserve(init.size());
    for (auto& entry : init) {
      _keys.emplace(std::move(entry.first, size() - 1);
      _values.emplace(std::move(entry.second);
    }
  }
  /*!
   * \brief Read element from OrderedDict.
   * \param key The key
   * \return the corresonding element.
   */
  const V& operator[](const K& key) const { 
    
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    return n == nullptr ? 0 : n->size();
  }
  /*! \return If OrderedDict contains the key */
  bool contains(const K& key) const {
    return find(key) != nullptr;
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
//   void Set(const K& key, const V& value) {
//     CopyOnWrite();
//     MapObj::InsertMaybeReHash(MapObj::KVType(key, value), &data_);
//   }
  /*! \return begin iterator */
  iterator begin() const { return iterator(_keys, _values); }
  /*! \return end iterator */
  iterator end() const { return iterator(_keys, _values); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(_->find(key)); }

//   void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*! \brief Iterator of the OrderedDict */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(std::move(kv.first), std::move(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

   private:
    iterator(const std::vector<std::pair<K, V>>::iterator& itr)  // NOLINT(*)
        : itr(itr) {}
    std::vector<std::pair<K, V>>::iterator itr;
  };

 private:
  /*! \brief Mapping from key to the index the value stored */
  std::unordered_map<K, index_t> _keys;
  /*! \brief Key-Value pair vector */
  std::vector<std::pair<K, V>> _values;
};

/*!
 * \brief Merge two Maps.
 * \param lhs the first Map to merge.
 * \param rhs the second Map to merge.
 * @return The merged Array. Original Maps are kept unchanged.
 */
template <typename K, typename V>
inline Map<K, V> Merge(Map<K, V> lhs, const Map<K, V>& rhs) {
  for (const auto& p : rhs) {
    lhs.Set(p.first, p.second);
  }
  return std::move(lhs);
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_COLLECTION_H_
