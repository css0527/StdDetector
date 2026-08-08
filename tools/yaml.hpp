#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>
#include <optional>

#include "logger.hpp"

namespace tools
{
inline YAML::Node load(const std::string & path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile & e) {
    logger()->error("[YAML] Failed to load file: {}", e.what());
    throw std::runtime_error("[YAML] Failed to load file: " + std::string(e.what()));
  } catch (const YAML::ParserException & e) {
    logger()->error("[YAML] Parser error: {}", e.what());
    throw std::runtime_error("[YAML] Parser error: " + std::string(e.what()));
  }
}

template <typename T>
inline T read(const YAML::Node & yaml, const std::string & key)
{
  if (yaml[key]) return yaml[key].as<T>();
  logger()->error("[YAML] {} not found!", key);
  throw std::runtime_error("[YAML] " + key + " not found!");
}

template <typename T>
inline std::optional<T> read_optional(const YAML::Node & yaml, const std::string & key)
{
  if (yaml[key]) return yaml[key].as<T>();
  return std::optional<T>();
}

}  // namespace tools

#endif  // TOOLS__YAML_HPP