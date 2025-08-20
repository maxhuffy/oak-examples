import { stringify } from 'safe-stable-stringify';

export const serializeToJson = stringify;

export const serializeToJsonOrThrow = (data: unknown): string => {
  const serialized = serializeToJson(data);
  if (serialized !== undefined) {
    return serialized;
  }
  throw new Error('Failed to serialize!');
};

export const parseJson = <T = unknown>(value: string): T => JSON.parse(value);

export const resolveReference = (jsonSchema: Record<string, any>, ref: string) => {
  const path = ref.substring(2).split('/');
  return path.reduce((schema, key) => schema[key], jsonSchema);
};

export const replaceRefs = (schema: Record<string, any>, jsonSchema: Record<string, any>) => {
  if (typeof schema === 'object' && schema !== null) {
    for (const key in schema) {
      if (schema[key] && typeof schema[key] === 'object') {
        if ('$ref' in schema[key]) {
          schema[key] = resolveReference(jsonSchema, schema[key]['$ref']);
        } else {
          replaceRefs(schema[key], jsonSchema);
        }
      } else if (Array.isArray(schema[key])) {
        for (const item of schema[key]) {
          if (typeof item === 'object' && item !== null) {
            replaceRefs(item, jsonSchema);
          }
        }
      }
    }
  }
  return schema;
};

export const replaceRefsInJsonSchema = (jsonSchema: Record<string, any>) => {
  return replaceRefs(jsonSchema, jsonSchema);
};
