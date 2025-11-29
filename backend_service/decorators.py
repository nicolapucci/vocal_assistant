from functools import wraps
from flask import request, jsonify, g
from typing import Callable, Any
from managers.postgres_manager import initialize_postgres_manager

postgres_manager = initialize_postgres_manager()

def device_endpoint(f:Callable)-> Callable:

    @wraps(f)
    def decorated_function(*args:Any,**kwargs: Any) -> Any:

        token = request.headers.get('X-Device-Token')

        if not token:
            return jsonify({
                'message':'Header mancante',
                'status':401
            }),401
        
        try:
            device = postgres_manager.get_device_by_token(token)
            if device is None:
                return jsonify({
                    'message':'Token non valido',
                    'status':401
                }),401
            if device.user_id is None:
                return jsonify({
                    'message':'Token non valido',
                    'status':403
                }),403
        except Exception as e:
            return jsonify({
                'message':'Errore interno',
                'status':500
            }),500

        print(f"DEBUG DECORATORE: Setto g.device per {device.id}. Proseguo con la view.")
        g.device = device

        return f(*args,**kwargs)
    return decorated_function